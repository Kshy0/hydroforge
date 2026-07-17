# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Kernel backend selection and adapter for hydroforge.

Set ``HYDROFORGE_BACKEND`` to choose the backend explicitly::

    export HYDROFORGE_BACKEND=metal    # Metal shaders (Apple Silicon)
    export HYDROFORGE_BACKEND=triton   # Triton JIT kernels (NVIDIA/AMD)
    export HYDROFORGE_BACKEND=cuda     # Compiled CUDA extensions (NVIDIA, or
                                       # AMD/ROCm via PyTorch's automatic hipify)
    export HYDROFORGE_BACKEND=torch    # Pure-PyTorch fallback

When unset, auto-detection picks the best available backend:
``triton`` → ``metal`` → ``torch``.  Triton covers both NVIDIA and AMD GPUs;
the ``cuda`` backend doubles as the AMD compiled path when explicitly selected.

When 'torch' is selected, :class:`KernelAdapter` wraps each PyTorch kernel
so it can be called with the unified hydroforge kwargs convention:

    kernel(arg_ptr=tensor, scalar_arg=value, BLOCK_SIZE=bs)

The adapter:
  1. Strips the ``_ptr`` suffix from tensor argument names.
  2. Drops unknown kwargs (``BLOCK_SIZE``, batch flags, etc.).
  3. Passes scalars directly (no buffer conversion needed).
"""

import os
from dataclasses import dataclass
from functools import cached_property
from typing import Any, Callable


def _resolve_backend() -> str:
    """Resolve kernel backend from HYDROFORGE_BACKEND environment variable.

    When the variable is unset, auto-detect in priority order:
    ``triton`` → ``metal`` → ``torch``.  Triton handles both NVIDIA and AMD
    GPUs, so AMD/ROCm users get Triton by default; the compiled ``cuda``
    backend (which PyTorch hipifies under ROCm) must be requested explicitly.
    """
    env = os.environ.get("HYDROFORGE_BACKEND", "").strip().lower()
    if env:
        return env
    try:
        import torch
        if torch.cuda.is_available():
            try:
                import triton  # noqa: F401
                return "triton"
            except ImportError:
                pass
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "metal"
    except ImportError:
        pass
    return "torch"


KERNEL_BACKEND: str = _resolve_backend()
os.environ.setdefault("HYDROFORGE_BACKEND", KERNEL_BACKEND)


def devices_match(actual, expected) -> bool:
    """Return whether a tensor device belongs to the expected model device."""
    import torch

    actual = torch.device(actual)
    expected = torch.device(expected)
    return (
        actual.type == expected.type
        and (expected.index is None or actual.index == expected.index)
    )


@dataclass(frozen=True)
class BackendRegistry:
    """Explicit lazy implementations of one logical kernel by backend."""

    implementations: dict[str, Callable[[], Callable]]
    name: str = "kernel"

    @cached_property
    def selected(self) -> Callable:
        return self.resolve()

    @property
    def available(self) -> tuple[str, ...]:
        return tuple(self.implementations)

    def resolve(self, backend: str | None = None) -> Callable:
        """Build the implementation for ``backend`` or the active backend."""
        backend = KERNEL_BACKEND if backend is None else backend
        try:
            factory = self.implementations[backend]
        except KeyError as exc:
            raise ValueError(
                f"Backend {backend!r} is not registered for {self.name}; "
                f"available={self.available}"
            ) from exc
        return factory()

    def __call__(self, **kwargs):
        return self.selected(**kwargs)

    def __getitem__(self, grid):
        return self.selected[grid]


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
        mapped_pos = {}
        for k in kwargs_keys:
            mapped = self._resolve_key(k)
            if mapped:
                if mapped in mapped_pos:
                    pairs[mapped_pos[mapped]] = (k, mapped)
                else:
                    mapped_pos[mapped] = len(pairs)
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


def make_batched_dispatcher(
    shared: Callable,
    batched: Callable,
    *,
    batch_key: str = "num_trials",
) -> Callable:
    """Select a shared or batched implementation without rewriting arguments.

    Grid construction belongs to the selected backend dispatcher.  This small
    adapter only expresses the semantic choice that the batched implementation
    is active when the model supplies a trial axis.
    """
    def dispatch(**kwargs: Any):
        trials = kwargs.get(batch_key)
        implementation = batched if trials is not None and trials > 1 else shared
        return implementation(**kwargs)

    return dispatch


# ── Triton dispatcher factory ─────────────────────────────────────────────

def _cdiv(n: int, d: int) -> int:
    return (n + d - 1) // d


def _validate_triton_optional_buffers(
    kw: dict[str, Any], optional_buffers: tuple[tuple[str, str], ...],
) -> None:
    """Require enabled feature buffers; disabled buffers remain ``None``.

    Triton specializes a kernel invocation containing ``None`` and removes the
    corresponding compile-time-disabled branch.  Passing the absence through
    directly avoids allocating or retaining temporary device tensors.
    """
    missing = [
        buffer for buffer, flag in optional_buffers
        if bool(kw.get(flag, False)) and kw.get(buffer) is None
    ]
    if missing:
        raise ValueError(
            "Enabled kernel feature is missing required buffer(s): "
            + ", ".join(missing)
        )


def make_triton_dispatcher(
    kernel,
    *,
    batched_kernel=None,
    size_key: str = "num_catchments",
    batched_grid: str = "parallel",
    batched_drop: tuple[str, ...] = (),
    non_batched_drop: tuple[str, ...] = (),
    optional_buffers: dict[str, str] | None = None,
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
    _optional_buffers = tuple((optional_buffers or {}).items())
    _non_batched_signature: tuple[str, ...] | None = None
    _non_batched_remove: tuple[str, ...] = ()

    def dispatch(**kw):
        nonlocal _non_batched_signature, _non_batched_remove
        bs = kw.pop("BLOCK_SIZE", 256)
        nt = kw.pop("num_trials", None)
        n = kw[size_key]
        if _optional_buffers:
            _validate_triton_optional_buffers(kw, _optional_buffers)
        if nt is not None and batched_kernel is not None:
            for k in batched_drop:
                kw.pop(k, None)
            if batched_grid == "parallel":
                grid = (_cdiv(n * nt, bs),)
            else:
                grid = (_cdiv(n, bs),)
            batched_kernel[grid](BLOCK_SIZE=bs, num_trials=nt, **kw)
        else:
            signature = tuple(kw)
            if signature != _non_batched_signature:
                _non_batched_signature = signature
                _non_batched_remove = tuple(
                    k for k in signature
                    if k.startswith("batched_") or k in non_batched_drop
                )
            for k in _non_batched_remove:
                kw.pop(k, None)
            grid = (_cdiv(n, bs),)
            kernel[grid](BLOCK_SIZE=bs, **kw)
    return dispatch


# ── Metal dispatcher factory ──────────────────────────────────────────────

_MISSING = object()

def make_metal_dispatcher(
    msl_source,
    kernel_name: str,
    args: tuple[str, ...],
    *,
    size_key: str | tuple[str, ...] = "num_catchments",
    template_vars: dict[str, str] | None = None,
    arg_defaults: dict[str, Any] | None = None,
    packed_args: dict[str, tuple[str, list[str]]] | None = None,
    function_constants: dict[str, int] | None = None,
    scalar_types: dict[str, str] | None = None,
    buffer_access: dict[str, str] | None = None,
    optional_buffers: dict[str, str] | None = None,
    group_size: int = 256,
) -> Callable:
    """Create a unified dispatch function for a Metal shader kernel.

    Parallel to :func:`make_triton_dispatcher`, backed by HydroForge's native
    Objective-C++ Metal launcher.

    The returned function accepts ``**kw`` and caches specialised Metal
    pipelines and packed argument buffers by their immutable inputs.

    Args:
        msl_source: MSL source code as a ``str``, **or** a ``pathlib.Path`` /
            ``os.PathLike`` pointing to a ``.metal`` file loaded at import time.
        kernel_name: Entry-point function name in the MSL source.
        args: Ordered tuple of kwarg names matching the fields of the kernel's
            argument-buffer struct. Nullable tensor fields are encoded as
            ``nil`` and must remain behind a disabled function-constant branch.
        size_key: One kwarg name providing the thread count, or multiple names
            whose integer values are multiplied to form the thread count.
        template_vars: ``{source_token: kwarg_key}`` for compile-time
            specialization.  The shader is recompiled and cached per unique
            combination of template values.
        arg_defaults: Default values for optional kwargs (e.g.
            ``{"HAS_RESERVOIR": False}``).  Missing kwargs without a default
            raise ``KeyError``.
        packed_args: ``{target_kwarg: (struct_fmt, [source_kwargs])}`` for
            packing multiple scalar kwargs into a single Metal struct buffer.
            Source kwargs are packed into an MPS ``uint8`` tensor and cached
            by their immutable values.
        function_constants: ``{kwarg_name: Metal function-constant index}``.
            Constants are specialised through ``MTLFunctionConstantValues``.
        scalar_types: ABI type for non-tensor arguments in the native launcher;
            supported values are ``"float32"`` and ``"int32"``.
        buffer_access: Explicit access mode for every tensor argument. Values
            are ``"read"``, ``"write"``, or ``"read_write"``; names and modes
            are forwarded to Metal resource-usage declarations.
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
    function_constants = function_constants or {}
    _optional_buffers = tuple((optional_buffers or {}).items())

    def _metal_literal(value: Any) -> str:
        """Render a Python specialization value as a valid MSL literal."""
        if value.__class__ is bool:
            return "true" if value else "false"
        if hasattr(value, "item"):
            value = value.item()
        return str(value)

    _pack_info = packed_args or {}
    _constant_keys = tuple(function_constants)
    if buffer_access is None:
        raise TypeError(
            f"Metal kernel {kernel_name!r} must declare buffer_access explicitly"
        )
    invalid_access = set(buffer_access.values()).difference(
        {"read", "write", "read_write"}
    )
    if invalid_access:
        raise ValueError(f"Invalid Metal buffer access modes: {sorted(invalid_access)}")
    _buffer_args = frozenset(buffer_access)
    unknown_buffers = _buffer_args.difference(args)
    if unknown_buffers:
        raise ValueError(
            f"Metal kernel {kernel_name!r} has unknown buffer_args: "
            f"{sorted(unknown_buffers)}"
        )
    scalar_kinds = scalar_types or {}
    missing_scalar_types = set(args).difference(_buffer_args, scalar_kinds)
    if missing_scalar_types:
        raise TypeError(
            f"Metal kernel {kernel_name!r} must declare scalar_types for: "
            f"{sorted(missing_scalar_types)}"
        )
    _native_types = tuple(
        "buffer" if name in _buffer_args else scalar_kinds[name]
        for name in args
    )
    _pipeline_cache: dict[tuple[Any, ...], int] = {}
    _native_packed_cache: dict[tuple[str, tuple[Any, ...]], Any] = {}
    _runtime: list[Any] = [None]
    from collections import OrderedDict
    _binding_cache: OrderedDict[tuple[int, tuple[Any, ...]], int] = OrderedDict()

    def prepare(**kw):
        native = _runtime[0]
        if native is None:
            from hydroforge.runtime.metal_kernel import load_metal_kernel

            native = load_metal_kernel()
            _runtime[0] = native
        for buffer, flag in _optional_buffers:
            if bool(kw.get(flag, _defaults.get(flag, False))) and kw.get(buffer) is None:
                raise ValueError(
                    f"Enabled kernel feature '{flag}' requires buffer '{buffer}'"
                )
        if _pack_info:
            import torch
            import struct as _struct_mod
            for target, (fmt, src_keys) in _pack_info.items():
                packed_values = tuple(
                    kw.get(name, _defaults.get(name)) for name in src_keys
                )
                packed_values = tuple(
                    value.item() if hasattr(value, "item") else value
                    for value in packed_values
                )
                packed_key = (target, packed_values)
                packed = _native_packed_cache.get(packed_key)
                if packed is None:
                    packed = torch.frombuffer(
                        bytearray(_struct_mod.pack(fmt, *packed_values)),
                        dtype=torch.uint8,
                    ).to("mps")
                    _native_packed_cache[packed_key] = packed
                kw[target] = packed
        values = []
        for name in args:
            value = kw.get(name, _defaults.get(name, _MISSING))
            if value is _MISSING:
                raise KeyError(name)
            values.append(value)

        key_values = []
        constants = []
        for name in _constant_keys:
            value = kw.get(name, _defaults.get(name, _MISSING))
            if value is _MISSING:
                raise KeyError(name)
            value = bool(value.item() if hasattr(value, "item") else value)
            key_values.append(value)
            constants.append((function_constants[name], value))
        template_values = []
        if template_vars:
            for name in template_vars.values():
                value = kw.get(name, _defaults.get(name, _MISSING))
                if value is _MISSING:
                    raise KeyError(name)
                if hasattr(value, "item"):
                    value = value.item()
                template_values.append(value)
        cache_key = (*template_values, *key_values)
        pipeline = _pipeline_cache.get(cache_key)
        if pipeline is None:
            specialized_source = source
            if template_vars:
                for (source_token, _), value in zip(
                    template_vars.items(), template_values, strict=True,
                ):
                    specialized_source = specialized_source.replace(
                        source_token, _metal_literal(value),
                    )
            pipeline = native.compile_pipeline(
                specialized_source, kernel_name, constants, _native_types,
                [buffer_access.get(name, "none") for name in args],
            )
            _pipeline_cache[cache_key] = pipeline
        binding_values = []
        for value in values:
            if hasattr(value, "data_ptr"):
                binding_values.append(
                    ("tensor", value.data_ptr(), value.storage_offset())
                )
            elif hasattr(value, "item"):
                binding_values.append(("scalar", value.item()))
            else:
                binding_values.append(("scalar", value))
        binding_key = (pipeline, tuple(binding_values))
        binding = _binding_cache.get(binding_key)
        if binding is None:
            binding = native.create_argument_binding(pipeline, values)
            _binding_cache[binding_key] = binding
            if len(_binding_cache) > 64:
                _, expired = _binding_cache.popitem(last=False)
                native.release_argument_binding(expired)
        else:
            _binding_cache.move_to_end(binding_key)
        if isinstance(size_key, str):
            threads = int(kw[size_key])
        else:
            threads = 1
            for key in size_key:
                threads *= int(kw[key])
        return native, pipeline, binding, threads, group_size

    def dispatch(**kw):
        prepared = prepare(**kw)
        from hydroforge.runtime.metal_kernel import recording_metal_sequence

        sequence = recording_metal_sequence()
        if sequence is not None:
            sequence.add_prepared(prepared, barrier=False)
            return
        native, pipeline, binding, threads, prepared_group_size = prepared
        native.dispatch(pipeline, binding, threads, prepared_group_size)

    dispatch.prepare = prepare

    return dispatch

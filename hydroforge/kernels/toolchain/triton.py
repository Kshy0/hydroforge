"""Triton toolchain: compile policy, launch options and precision variants.

Compile policy
    Physics kernels are large fused 1-D programs. Two Triton passes spend most
    of their compile time producing results those programs cannot use, so
    HydroForge skips them where the skip provably leaves the generated code
    unchanged:

    - ``TritonGPUCoalesce`` recomputes a whole-kernel data-flow slice for
      every memory access (roughly quadratic in kernel size; about 99% of the
      compile time of VIC's largest kernels). It only chooses per-thread vector
      widths for memory accesses; when every pointer tensor is 1-D with no more
      elements than the block has threads, the only choice is the default
      layout.
    - Software pipelining (``num_stages``) only transforms loads feeding dot or
      tensor-descriptor operations; without them ``num_stages=1`` yields the
      same code and skips the latency analysis.

    Both decisions are made per kernel from its TTIR. :func:`activate`
    installs the compiler hooks once and :func:`adopt` marks a HydroForge
    kernel; the policy is active only inside a marked kernel's compilation
    (also on asynchronous compile threads) and is part of its disk-cache key,
    so launches pay nothing for it. Unmarked kernels (Inductor, user Triton)
    compile unchanged; so does everything when the required compiler APIs are
    absent or the target is not NVIDIA.
Launch options
    :func:`launch_options` resolves a kernel's math options once: physics
    kernels follow ``HYDROFORGE_FAST_MATH``, framework kernels keep IEEE
    subnormals.
Precision variants
    Triton treats an unannotated Python ``float`` as fp32. A precision-
    dependent scalar is lowered to a typed runtime scalar by rebuilding the
    kernel with explicit annotations (:func:`precision_variant`); a compound
    program's inner launches find the resolved ABI through
    :func:`triton_precision_context`.
"""

from __future__ import annotations

import dataclasses
import inspect
import logging
import re
import threading
import types
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from functools import cache, wraps
from pathlib import Path
from typing import Any

from hydroforge.kernels.codegen.types import SCALARS
from hydroforge.kernels.toolchain import CompileRequest
from hydroforge.platform.backend import TRITON

_POLICY = "hydroforge-physics-v3-fallback"
_logger = logging.getLogger(__name__)
_native_specializations: set[str] = set()
_policy_attempt: ContextVar[bool | None] = ContextVar(
    "hydroforge_triton_policy_attempt", default=None
)
_MEMORY_OP = re.compile(r"\btt\.(?:load|store|atomic_rmw|atomic_cas)\b")
_POINTER_TENSOR = re.compile(r"tensor<([0-9x]+)x!tt\.ptr<")
_PIPELINED_OPS = re.compile(r"\btt\.(?:dot|dot_scaled)\b|\btt\.descriptor_|\bttng\.")
_state = threading.local()
_active_policy: ContextVar[bool] = ContextVar(
    "hydroforge_triton_compile_policy", default=False
)
_install_lock = threading.Lock()


def _coalesce_is_identity(ttir: str, threads: int) -> bool:
    """Whether every memory access is 1-D and no wider than the block."""

    for line in ttir.splitlines():
        if not _MEMORY_OP.search(line):
            continue
        shapes = _POINTER_TENSOR.findall(line)
        if not shapes:
            return False
        for shape in shapes:
            dims = shape.split("x")
            if len(dims) != 1 or int(dims[0]) > threads:
                return False
    return True


def _compile_with_fallback(compile_fn):
    """Compile under the policy; retry a policy failure natively from the AST.

    Only compilation is routed; JIT execution stays outside the retry boundary.
    """

    @wraps(compile_fn)
    def compile_kernel(src, target=None, options=None, _env_vars=None):
        from triton import knobs
        from triton.compiler.compiler import (
            ASTSource,
            get_cache_invalidating_env_vars,
            get_cache_key,
            make_backend,
        )

        if not isinstance(src, ASTSource) or target is None or target.backend != "cuda":
            return compile_fn(src, target=target, options=options, _env_vars=_env_vars)

        env = get_cache_invalidating_env_vars() if _env_vars is None else _env_vars
        backend = make_backend(target)
        parsed = backend.parse_options(dict(options or {}, **src.parse_options()))
        hook = knobs.runtime.add_stages_inspection_hook
        preceding_key = hook() if hook is not None else ("", "")
        key = get_cache_key(src, backend, parsed, env) + repr(preceding_key)
        if key in _native_specializations:
            return compile_fn(src, target=target, options=options, _env_vars=env)
        policy = _active_policy.set(True)
        attempt = _policy_attempt.set(False)
        try:
            try:
                return compile_fn(src, target=target, options=options, _env_vars=env)
            except Exception:
                if not _policy_attempt.get():
                    raise
                # Re-enter the full compiler from the AST, never mutated IR or
                # metadata. Native compilation has its own Triton disk-cache key.
                native = _active_policy.set(False)
                try:
                    result = compile_fn(
                        src, target=target, options=options, _env_vars=env
                    )
                finally:
                    _active_policy.reset(native)
                _native_specializations.add(key)
                _logger.warning(
                    "Triton policy compilation failed for %s; using the native pipeline",
                    src.name,
                    exc_info=True,
                )
                return result
        finally:
            _policy_attempt.reset(attempt)
            _active_policy.reset(policy)

    compile_kernel.hydroforge_fallback = True
    return compile_kernel


def _install() -> bool:
    try:
        from triton import knobs
        from triton._C.libtriton import passes
        from triton.backends.compiler import Language
        from triton.runtime.jit import JITFunction
    except ImportError:
        return False
    if not hasattr(knobs.runtime, "add_stages_inspection_hook"):
        return False
    add_coalesce = getattr(getattr(passes, "ttgpuir", None), "add_coalesce", None)
    original_compile = getattr(JITFunction, "_do_compile", None)
    if not callable(add_coalesce) or not callable(original_compile):
        return False
    if getattr(add_coalesce, "hydroforge_policy", False):
        return True

    @wraps(original_compile)
    def do_compile(kernel, *args, **kwargs):
        # A kernel's binder (re)assigns its compiler callable, so adopted
        # kernels adapt it at the cache-miss boundary, on the launching
        # thread; the policy itself is active inside that compilation only.
        if getattr(kernel, "hydroforge_policy", False) and not getattr(
            kernel.compile, "hydroforge_fallback", False
        ):
            with _install_lock:
                if not getattr(kernel.compile, "hydroforge_fallback", False):
                    kernel.compile = _compile_with_fallback(kernel.compile)
        return original_compile(kernel, *args, **kwargs)

    JITFunction._do_compile = do_compile

    def conditional_coalesce(pm) -> None:
        if not getattr(_state, "skip_coalesce", False):
            add_coalesce(pm)

    conditional_coalesce.hydroforge_policy = True
    passes.ttgpuir.add_coalesce = conditional_coalesce
    previous = knobs.runtime.add_stages_inspection_hook

    def inspect_stages(
        backend=None, stages=None, options=None, language=None, capability=None
    ):
        if backend is None:
            # Cache-key query: this policy changes which passes run.
            key, digest = previous() if previous is not None else ("", "")
            if not _active_policy.get():
                return key, digest
            return f"{key}{_POLICY}", f"{digest}{_POLICY}"
        if previous is not None:
            previous(backend, stages, options, language, capability)
        if not _active_policy.get():
            return
        make_ttgir = stages.get("ttgir")
        # Verified on NVIDIA only; AMD passes no capability.
        if language != Language.TRITON or make_ttgir is None or capability is None:
            return

        def policy_ttgir(src, metadata):
            if _policy_attempt.get() is None:
                return make_ttgir(src, metadata)
            ttir = str(src)
            if any(
                _MEMORY_OP.search(line) and not _POINTER_TENSOR.search(line)
                for line in ttir.splitlines()
            ):
                return make_ttgir(src, metadata)
            threads = getattr(options, "warp_size", 32) * options.num_warps
            compile_ttgir = make_ttgir
            skip_coalesce = _coalesce_is_identity(ttir, threads)
            single_stage = (
                options.num_stages != 1 and _PIPELINED_OPS.search(ttir) is None
            )
            _policy_attempt.set(skip_coalesce or single_stage)
            if single_stage:
                one_stage = dataclasses.replace(options, num_stages=1)

                def compile_ttgir(module, data):
                    return backend.make_ttgir(module, data, one_stage, capability)

            previous_skip = getattr(_state, "skip_coalesce", False)
            _state.skip_coalesce = skip_coalesce
            try:
                return compile_ttgir(src, metadata)
            finally:
                _state.skip_coalesce = previous_skip

        stages["ttgir"] = policy_ttgir

    knobs.runtime.add_stages_inspection_hook = inspect_stages
    return True


def activate() -> bool:
    """Install the stages hook of the compile policy once; report availability."""

    with _install_lock:
        return _install()


def adopt(kernel: Any) -> Any:
    """Compile ``kernel`` under the HydroForge policy from now on."""

    if hasattr(kernel, "params") and activate():
        kernel.hydroforge_policy = True
    return kernel


@cache
def _toolkit_libdevice() -> str | None:
    from torch.utils.cpp_extension import CUDA_HOME, include_paths

    if CUDA_HOME is None:
        return None
    for root in include_paths("cuda"):
        candidate = Path(root).parent / "nvvm" / "libdevice" / "libdevice.10.bc"
        if candidate.is_file():
            return str(candidate)
    return None


_LAUNCH_OPTIONS: dict[tuple[int, bool], tuple[Any, Any, Mapping[str, Any]]] = {}


def launch_options(kernel: Any, *, physics: bool) -> Mapping[str, Any]:
    """Math options of one JIT kernel, resolved once per active driver.

    Physics kernels follow the math mode; framework kernels always preserve
    subnormals and never contract multiplies and adds, so their results do
    not depend on how the generated code is structured. Already lowered
    launch adapters have no JIT parameters or compiler options. Resolving a
    kernel's options adopts it into the compile policy.
    """

    if not hasattr(kernel, "params"):
        return {}
    from triton.runtime import driver

    active = driver.active
    key = (id(kernel), physics)
    cached = _LAUNCH_OPTIONS.get(key)
    if cached is not None and cached[0] is kernel and cached[1] is active:
        return cached[2]
    adopt(kernel)
    options: dict[str, Any] = {} if physics else {"enable_fp_fusion": False}
    target = active.get_current_target().backend
    if target == "cpu":
        fast = physics and TRITON.math.physics_fast_math
        options.update(enable_fast_math=fast, enable_fp_fusion=fast)
    if target == "cuda":
        from triton import knobs

        options["enable_reflect_ftz"] = physics and TRITON.math.physics_fast_math
        if not knobs.nvidia.libdevice_path:
            libdevice = _toolkit_libdevice()
            if libdevice is not None:
                options["extern_libs"] = (("libdevice", libdevice),)
    _LAUNCH_OPTIONS[key] = (kernel, active, options)
    return options


def warmup_request(build: Callable[[], object], *, cost: int = 0) -> CompileRequest:
    """Request compiling what ``build`` warms up with exact launch arguments.

    Triton deduplicates its compilations itself, so every warmup is its own
    request.
    """

    return CompileRequest(TRITON.toolchain, f"triton:{id(build)}", cost, build)


_WARMING_UP: ContextVar[bool] = ContextVar("hydroforge_triton_warmup", default=False)


@contextmanager
def warming_up():
    """Compile instead of launch every :func:`launch_triton_kernel` call."""

    token = _WARMING_UP.set(True)
    try:
        yield
    finally:
        _WARMING_UP.reset(token)


def warming() -> bool:
    return _WARMING_UP.get()


_ACTIVE_TRITON_PRECISION: ContextVar[
    tuple[str | None, frozenset[str], Mapping[str, str]] | None
] = ContextVar(
    "hydroforge_triton_precision",
    default=None,
)


@contextmanager
def triton_precision_context(
    precision: str | None,
    scalar_names: frozenset[str] = frozenset(),
    *,
    scalar_types: Mapping[str, str] | None = None,
):
    """Expose one resolved Triton scalar ABI while a program is launching.

    Compound programs are allowed to contain ordinary Python launch helpers,
    so their inner Triton kernels cannot receive the logical ``KernelSpec``
    through the normal factory context.  This small context carries only the
    resolved floating-point ABI needed by ``launch_triton_kernel``.
    """

    token = _ACTIVE_TRITON_PRECISION.set((precision, scalar_names, scalar_types or {}))
    try:
        yield
    finally:
        _ACTIVE_TRITON_PRECISION.reset(token)


def active_triton_precision() -> tuple[str | None, frozenset[str]] | None:
    """Return the active compound-program Triton precision contract."""

    active = _ACTIVE_TRITON_PRECISION.get()
    return None if active is None else active[:2]


_TRITON_PRECISION_VARIANTS: dict[
    tuple[int, tuple[tuple[str, str], ...]], tuple[Any, Any]
] = {}


def precision_variant(
    kernel: Any,
    precision: str | None,
    scalar_names: frozenset[str] = frozenset(),
    *,
    scalar_types: Mapping[str, str] | None = None,
) -> Any:
    """Return a Triton JIT variant with the resolved scalar annotations.

    Triton treats an unannotated Python ``float`` as fp32 even when the
    surrounding model is fp64.  Rebuilding the small Python function with
    precision-specific annotations lets Triton generate an ABI-correct
    specialization without changing every downstream launch helper.  Mock
    kernels used by contract tests intentionally do not support this and are
    validated strictly instead.
    """

    types_by_name = {
        **({name: precision for name in scalar_names} if precision is not None else {}),
        **(scalar_types or {}),
    }
    if precision is not None and precision not in {"float32", "float64"}:
        raise ValueError("Triton precision must be float32 or float64")
    if any(kind not in {"float32", "float64"} for kind in types_by_name.values()):
        raise ValueError("Triton scalar precision must be float32 or float64")
    if not types_by_name or not hasattr(kernel, "params") or not hasattr(kernel, "fn"):
        return kernel
    parameters = {parameter.name: parameter for parameter in kernel.params}
    if unknown := set(types_by_name).difference(parameters):
        raise ValueError(f"unknown Triton scalar parameters: {sorted(unknown)}")
    types_by_name = {
        name: kind for name, kind in types_by_name.items() if name in parameters
    }
    if not types_by_name:
        return kernel
    # A precision-dependent parameter is deliberately lowered to a typed
    # runtime scalar.  Matching the dtype alone is not sufficient: a kernel
    # may already spell the parameter as ``tl.float64`` while retaining the
    # ``tl.constexpr`` qualifier, which would still make Triton specialize the
    # value as a compile-time constant and bypass the model precision contract.
    if all(
        parameters[name].annotation_type == SCALARS[kind].triton
        and not getattr(parameters[name], "is_constexpr", False)
        for name, kind in types_by_name.items()
    ):
        return kernel

    # Keep the source object alongside the integer key.  This avoids invoking
    # Triton's relatively expensive ``JITFunction.__hash__`` while also
    # preventing an id-reuse collision after a lazily-created implementation is
    # collected.
    typed_names = tuple(sorted(types_by_name.items()))
    key = (id(kernel), typed_names)
    cached = _TRITON_PRECISION_VARIANTS.get(key)
    if cached is not None:
        source_kernel, variant = cached
        if source_kernel is kernel:
            return variant

    import triton
    import triton.language as tl

    source = kernel.fn
    annotations = dict(getattr(source, "__annotations__", {}))
    for name, kind in typed_names:
        annotations[name] = tl.float32 if kind == "float32" else tl.float64
    suffix = "_".join(f"{name}_{kind}" for name, kind in typed_names)
    variant_name = f"{source.__name__}__hydroforge_{suffix}"
    variant = types.FunctionType(
        source.__code__,
        source.__globals__,
        variant_name,
        source.__defaults__,
        source.__closure__,
    )
    variant.__kwdefaults__ = source.__kwdefaults__
    variant.__annotations__ = annotations
    variant.__dict__.update(getattr(source, "__dict__", {}))
    variant.__doc__ = source.__doc__
    variant.__module__ = source.__module__
    variant.__qualname__ = f"{source.__qualname__}__hydroforge_{suffix}"
    jit_kwargs = {
        name: getattr(kernel, name)
        for name in (
            "version",
            "do_not_specialize",
            "do_not_specialize_on_alignment",
            "debug",
            "noinline",
            "launch_metadata",
        )
        if hasattr(kernel, name) and getattr(kernel, name) not in (None, [], {})
    }
    compiled = triton.jit(variant, **jit_kwargs)
    _TRITON_PRECISION_VARIANTS[key] = (kernel, compiled)
    return compiled


@dataclass(frozen=True, slots=True)
class _TritonLaunchSignature:
    """Launch-argument layout of one kernel, inspected once per process."""

    signature: inspect.Signature
    # ``None`` when the signature has kinds that need full ``bind_partial``.
    positional: tuple[str, ...] | None
    float_defaults: frozenset[str]


_TRITON_LAUNCH_SIGNATURES: dict[int, tuple[Any, _TritonLaunchSignature | None]] = {}


_TRITON_LAUNCH_VARIANTS: dict[
    tuple[int, str | None, frozenset[str], tuple[tuple[str, str], ...]], tuple[Any, Any]
] = {}


def _triton_launch_signature(kernel: Any) -> _TritonLaunchSignature | None:
    cached = _TRITON_LAUNCH_SIGNATURES.get(id(kernel))
    if cached is not None and cached[0] is kernel:
        return cached[1]
    try:
        signature = inspect.signature(getattr(kernel, "fn", kernel))
    except (TypeError, ValueError):
        layout = None
    else:
        parameters = tuple(signature.parameters.values())
        simple = all(
            parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
            for parameter in parameters
        )
        layout = _TritonLaunchSignature(
            signature,
            tuple(parameter.name for parameter in parameters) if simple else None,
            frozenset(
                parameter.name
                for parameter in parameters
                if isinstance(parameter.default, float)
            ),
        )
    _TRITON_LAUNCH_SIGNATURES[id(kernel)] = (kernel, layout)
    return layout


def _declared_float_arguments(
    layout: _TritonLaunchSignature | None,
    declared: frozenset[str],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> frozenset[str]:
    """Return declared names bound to Python floats, as ``bind_partial`` would."""

    if layout is None:
        return frozenset()
    parameters = layout.signature.parameters
    positional = layout.positional
    if positional is None:
        try:
            bound = layout.signature.bind_partial(
                *args,
                **{name: value for name, value in kwargs.items() if name in parameters},
            )
            bound.apply_defaults()
        except (TypeError, ValueError):
            return frozenset()
        return frozenset(
            name
            for name, value in bound.arguments.items()
            if name in declared and isinstance(value, float)
        )
    if len(args) > len(positional) or any(
        name in kwargs for name in positional[: len(args)]
    ):
        return frozenset()
    names = {
        name
        for name, value in zip(positional, args)
        if name in declared and isinstance(value, float)
    }
    names.update(
        name
        for name, value in kwargs.items()
        if name in declared and name in parameters and isinstance(value, float)
    )
    names.update(
        name
        for name in layout.float_defaults.intersection(declared)
        if name not in kwargs and name not in positional[: len(args)]
    )
    return frozenset(names)


def launch_variant(
    kernel: Any, args: tuple[Any, ...], kwargs: Mapping[str, Any]
) -> Any:
    """Return ``kernel``, or its variant for the active program precision.

    The launch arguments are bound first to find the declared floating scalar
    parameters they pass; the variant is cached per precision and names.
    """

    active = _ACTIVE_TRITON_PRECISION.get()
    if active is None:
        return kernel
    precision, declared, scalar_types = active
    names = _declared_float_arguments(
        _triton_launch_signature(kernel),
        declared | frozenset(scalar_types),
        args,
        dict(kwargs),
    )
    typed = tuple(
        sorted((name, kind) for name, kind in scalar_types.items() if name in names)
    )
    key = (id(kernel), precision, names, typed)
    cached = _TRITON_LAUNCH_VARIANTS.get(key)
    if cached is not None and cached[0] is kernel:
        return cached[1]
    selected = (
        precision_variant(kernel, precision, names, scalar_types=dict(typed))
        if typed
        else precision_variant(kernel, precision, names)
    )
    _TRITON_LAUNCH_VARIANTS[key] = (kernel, selected)
    return selected

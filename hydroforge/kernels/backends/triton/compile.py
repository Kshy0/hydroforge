"""Triton compile-time policy for physics kernels.

Physics kernels are large fused 1-D programs. Two Triton passes spend most of
their compile time producing results those programs cannot use, so HydroForge
skips them where the skip provably leaves the generated code unchanged:

- ``TritonGPUCoalesce`` recomputes a whole-kernel data-flow slice for every
  memory access (roughly quadratic in kernel size; about 99% of the compile
  time of VIC's largest kernels). It only chooses per-thread vector widths for
  memory accesses; when every pointer tensor is 1-D with no more elements than
  the block has threads, the only choice is the default layout.
- Software pipelining (``num_stages``) only transforms loads feeding dot or
  tensor-descriptor operations; without them ``num_stages=1`` yields the same
  code and skips the latency analysis.

Both decisions are made per kernel from its TTIR, and the policy is part of
Triton's cache key. HydroForge launch and precompile scopes enable the policy
for NVIDIA targets when the required compilation APIs are available; elsewhere
Triton compiles unchanged. :func:`precompile` compiles an operator program's
Triton specializations concurrently before their first launch.
"""

from __future__ import annotations

import dataclasses
import logging
import re
import threading
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from functools import wraps

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
    """Route compilation only; JIT execution stays outside the retry boundary."""

    @wraps(compile_fn)
    def compile_kernel(src, target=None, options=None, _env_vars=None):
        from triton import knobs
        from triton.compiler.compiler import (
            ASTSource,
            get_cache_invalidating_env_vars,
            get_cache_key,
            make_backend,
        )

        if (
            not _active_policy.get()
            or not isinstance(src, ASTSource)
            or target is None
            or target.backend != "cuda"
        ):
            return compile_fn(src, target=target, options=options, _env_vars=_env_vars)

        env = get_cache_invalidating_env_vars() if _env_vars is None else _env_vars
        backend = make_backend(target)
        parsed = backend.parse_options(dict(options or {}, **src.parse_options()))
        native_token = _active_policy.set(False)
        try:
            hook = knobs.runtime.add_stages_inspection_hook
            preceding_key = hook() if hook is not None else ("", "")
        finally:
            _active_policy.reset(native_token)
        key = get_cache_key(src, backend, parsed, env) + repr(preceding_key)

        def native_compile():
            token = _active_policy.set(False)
            try:
                return compile_fn(src, target=target, options=options, _env_vars=env)
            finally:
                _active_policy.reset(token)

        if key in _native_specializations:
            return native_compile()
        token = _policy_attempt.set(False)
        try:
            try:
                return compile_fn(src, target=target, options=options, _env_vars=env)
            except Exception:
                if not _policy_attempt.get():
                    raise
                # Re-enter the full compiler from the AST, never mutated IR or
                # metadata. Native compilation has its own Triton disk-cache key.
                result = native_compile()
                _native_specializations.add(key)
                _logger.warning(
                    "Triton policy compilation failed for %s; using the native pipeline",
                    src.name,
                    exc_info=True,
                )
                return result
        finally:
            _policy_attempt.reset(token)

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
        # Existing binders may already hold the compiler callable. Adapt at the
        # cache-miss boundary, including compilation submitted to worker threads.
        if _active_policy.get():
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


@contextmanager
def compilation_policy():
    """Enable HydroForge passes only for the current launch or compilation.

    Triton exposes a process-wide stages hook. Install its routing shim lazily;
    unrelated threads and launches retain the preceding hook's pipeline and key.
    """
    with _install_lock:
        enabled = _install()
    token = _active_policy.set(enabled)
    try:
        yield
    finally:
        _active_policy.reset(token)


def precompile(warmups: Iterable[Callable[[], object]]) -> None:
    """Compile Triton specializations concurrently before their first launch.

    Each warmup calls ``kernel.warmup`` with the exact launch arguments, so the
    compiled variants are the ones the launches use. Triton compiles one kernel
    on one thread; distinct kernels compile in parallel
    (``HYDROFORGE_PRECOMPILE_JOBS``, as for runtime-compiled CUDA).
    """

    pending = list(warmups)
    if not pending:
        return
    from triton.runtime._async_compile import AsyncCompileMode

    from hydroforge.kernels.backends.precompile import precompile_jobs

    jobs = precompile_jobs(len(pending), default_jobs=6)
    with compilation_policy():
        if jobs == 1:
            for warmup in pending:
                warmup()
            return
        enabled = _active_policy.get()
        with (
            ThreadPoolExecutor(
                max_workers=jobs,
                initializer=lambda: _active_policy.set(enabled),
            ) as pool,
            AsyncCompileMode(pool),
        ):
            for warmup in pending:
                warmup()


__all__ = ["precompile"]

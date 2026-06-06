# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""
Generic CUDA kernel loader and Triton-compatible wrapper for hydroforge.

Provides:
  - :func:`load_cu_module` — compile ``.cu`` files via ``load_inline`` into
    a pybind11 extension, caching the result.
  - :class:`CudaKernel` — generic wrapper that adapts a CUDA launcher to the
    Triton ``kernel[grid](**kwargs)`` calling convention, eliminating the need
    for per-kernel wrapper classes.

Example
-------
::

    from hydroforge.runtime.cuda_kernel import CudaKernel, load_cu_module

    mod = load_cu_module("cmfgpu_physics", [physics_cu, launchers_cu])

    compute_outflow_kernel = CudaKernel(
        mod, "launch_outflow",
        args=[
            "downstream_idx_ptr", "river_inflow_ptr", "river_outflow_ptr",
            ...
            ("gravity", float), ("time_step", float),
            ("num_catchments", int),
            ("HAS_BIFURCATION", bool, True), ("HAS_RESERVOIR", bool, False),
        ],
        nullable={"is_dam_upstream_ptr": "torch.bool"},
    )

    # Now usable with Triton calling convention:
    compute_outflow_kernel[grid](river_depth_ptr=tensor, gravity=9.81, ...)
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.util
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

_module_cache: Dict[str, Any] = {}

_MANIFEST_FILENAME = "compile_manifest.json"


def _safe_path_component(value: str) -> str:
    component = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    return component[:80] or "unknown"


def _host_build_scope() -> str:
    return _safe_path_component(socket.gethostname() or "host")


def load_build_manifest(build_dir: Union[str, Path]) -> dict:
    """Load the compile manifest from a build directory."""
    p = Path(build_dir) / _MANIFEST_FILENAME
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def update_build_manifest(
    build_dir: Union[str, Path], section: str, data: dict
) -> None:
    """Update one section of the compile manifest and write it back."""
    manifest = load_build_manifest(build_dir)
    manifest[section] = data
    p = Path(build_dir) / _MANIFEST_FILENAME
    with open(p, "w") as f:
        json.dump(manifest, f, indent=2)


def check_build_manifest(
    build_dir: Union[str, Path], section: str, expected_name: str
) -> None:
    """Warn if a precompiled kernel in *build_dir* doesn't match current config.

    Called before compilation so the user knows the precompiled cache won't
    be reused and a slow recompilation is about to happen.
    """
    manifest = load_build_manifest(build_dir)
    entry = manifest.get(section)
    if entry is None:
        return  # No prior manifest entry → first compilation, nothing to check
    cached_name = entry.get("module_name", "")
    if cached_name != expected_name:
        import warnings
        warnings.warn(
            f"Precompiled {section} kernel mismatch in {build_dir}:\n"
            f"  precompiled: {cached_name}\n"
            f"  current:     {expected_name}\n"
            f"Re-run precompile to update the cache.",
            stacklevel=3,
        )


def load_cu_module(
    name: str,
    cuda_files: Sequence[Union[str, Path]],
    *,
    extra_cuda_cflags: Sequence[str] = ("-O3", "--use_fast_math"),
    verbose: bool = False,
    build_directory: Optional[Union[str, Path]] = None,
) -> Any:
    """Compile ``.cu`` files into a pybind11 extension module (cached).

    Args:
        name: Unique module name for caching.
        cuda_files: Paths to ``.cu`` / ``.cpp`` source files.
        extra_cuda_cflags: Extra NVCC flags.
        verbose: Print compilation output.
        build_directory: If given, compiled artefacts (``.so``, ``.o``) are
            stored in this directory so they can be reused across runs.

    Returns:
        The compiled extension module.
    """
    if name in _module_cache:
        return _module_cache[name]

    from torch.utils.cpp_extension import load

    sources = [str(Path(f).resolve()) for f in cuda_files]
    _verbose = verbose or os.environ.get("HYDROFORGE_CUDA_VERBOSE", "") == "1"

    kw: Dict[str, Any] = dict(
        name=name,
        sources=sources,
        extra_cuda_cflags=list(extra_cuda_cflags),
        verbose=_verbose,
    )
    if build_directory is not None:
        bd = Path(build_directory)
        bd.mkdir(parents=True, exist_ok=True)
        kw["build_directory"] = str(bd)

    mod = load(**kw)
    _module_cache[name] = mod
    return mod


def _env_truthy(name: str, default: bool = False) -> bool:
    val = os.environ.get(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _normalise_inline_source(src: Union[str, Sequence[str]]) -> str:
    if isinstance(src, str):
        return src
    return "\n".join(src)


def _normalise_hip_cflags(flags: Sequence[str]) -> List[str]:
    """Rename the one nvcc-only flag hipcc/clang spells differently.

    PyTorch's JIT ``load_inline`` auto-hipifies the *source* (includes,
    ``cudaStream_t``, ``<<<>>>`` launches, …) but does not translate device
    *compile flags*, and clang-based hipcc rejects nvcc's ``--use_fast_math``.
    The C++ symbols need no shim: ``c10::cuda::getCurrentCUDAStream()`` resolves
    natively on ROCm because PyTorch keeps the ``c10::cuda`` namespace alive in
    its hipified stream headers (see Note [Masquerading as CUDA]).
    """
    return ["-ffast-math" if f == "--use_fast_math" else f for f in flags]


def _visible_device_arches() -> List[str]:
    # An explicit arch list pins the build target, so the live device set is
    # irrelevant to the cache key.  ROCm honours PYTORCH_ROCM_ARCH; CUDA uses
    # TORCH_CUDA_ARCH_LIST.
    if torch.version.hip is not None:
        if os.environ.get("PYTORCH_ROCM_ARCH"):
            return []
    elif os.environ.get("TORCH_CUDA_ARCH_LIST"):
        return []
    try:
        if not torch.cuda.is_available():
            return []
        arches = set()
        for idx in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(idx)
            gcn_arch = getattr(props, "gcnArchName", None)
            if gcn_arch:
                arches.add(f"gcn:{gcn_arch}")
            else:
                arches.add(f"sm_{props.major}{props.minor}")
        return sorted(arches)
    except Exception:
        return []


def _build_fingerprint(
    *,
    name: str,
    cpp_sources: Union[str, Sequence[str]],
    cuda_sources: Union[str, Sequence[str]],
    functions: Sequence[str],
    extra_cuda_cflags: Sequence[str],
) -> dict:
    from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

    return {
        "module_name": name,
        "toolchain": "hip" if torch.version.hip is not None else "cuda",
        "cpp_sha256": hashlib.sha256(_normalise_inline_source(cpp_sources).encode()).hexdigest(),
        "cuda_sha256": hashlib.sha256(_normalise_inline_source(cuda_sources).encode()).hexdigest(),
        "functions": list(functions),
        "extra_cuda_cflags": list(extra_cuda_cflags),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "torch_hip": torch.version.hip,
        "cuda_home": str(CUDA_HOME) if CUDA_HOME is not None else None,
        "rocm_home": str(ROCM_HOME) if ROCM_HOME is not None else None,
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "pytorch_rocm_arch": os.environ.get("PYTORCH_ROCM_ARCH"),
        "visible_device_arches": _visible_device_arches(),
        "cc": os.environ.get("CC"),
        "cxx": os.environ.get("CXX"),
        "cuda_host_cxx": os.environ.get("CUDAHOSTCXX"),
    }


def _fingerprint_digest(fingerprint: dict) -> str:
    payload = json.dumps(fingerprint, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()


def _maybe_use_system_compiler(mode: str = "auto") -> Dict[str, Optional[str]]:
    """Temporarily switch host compilers to system gcc/g++ when requested.

    Some conda compiler wrappers ship sysroot linker scripts that make PyTorch
    CUDA extension linking fail even though ``nvcc`` itself is present.  In
    ``auto`` mode we only switch when CC/CXX look like conda compiler wrappers.
    """
    mode = mode.strip().lower()
    if mode in {"0", "false", "no", "off", "never"}:
        return {}

    cc = os.environ.get("CC", "")
    cxx = os.environ.get("CXX", "")
    if mode == "auto" and "conda-linux-gnu" not in (cc + cxx):
        return {}

    gcc = shutil.which("gcc")
    gxx = shutil.which("g++")
    if not gcc or not gxx:
        return {}

    old = {"CC": os.environ.get("CC"), "CXX": os.environ.get("CXX")}
    os.environ["CC"] = gcc
    os.environ["CXX"] = gxx
    return old


def _restore_compiler_env(old: Dict[str, Optional[str]]) -> None:
    for key, val in old.items():
        if val is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = val


def _import_extension_from_so(name: str, so_path: Path) -> Any:
    import torch  # noqa: F401 - ensure torch shared libraries are loaded first

    spec = importlib.util.spec_from_file_location(name, str(so_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create extension spec for {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _env_float(name: str, default: float) -> float:
    val = os.environ.get(name)
    if val is None:
        return default
    try:
        return float(val)
    except ValueError:
        return default


def _maybe_clear_stale_torch_lock(
    build_dir: Path,
    *,
    env_prefix: str,
    env_kind: str,
    verbose: bool,
) -> None:
    """Remove abandoned PyTorch extension build locks.

    ``torch.utils.cpp_extension`` uses a lock file named ``lock`` in the build
    directory and waits indefinitely when it already exists.  If a previous
    Python process is interrupted while compiling, that empty lock can be left
    behind and every later rebuild will appear to hang before ninja starts.
    """
    lock_path = build_dir / "lock"
    if not lock_path.exists():
        return

    stale_after = _env_float(
        f"{env_prefix}_{env_kind}_LOCK_STALE_SECONDS",
        _env_float(f"HYDROFORGE_{env_kind}_LOCK_STALE_SECONDS", 900.0),
    )
    if stale_after <= 0.0:
        return

    try:
        age = time.time() - lock_path.stat().st_mtime
    except OSError:
        return
    if age < stale_after:
        return

    try:
        lock_path.unlink()
    except FileNotFoundError:
        return
    if verbose:
        print(
            f"[hydroforge] removed stale {env_kind} extension lock "
            f"{lock_path} (age {age:.0f}s)",
            file=sys.stderr,
        )


def load_inline_cu_module(
    name: str,
    *,
    cpp_sources: Union[str, Sequence[str]],
    cuda_sources: Union[str, Sequence[str]],
    functions: Sequence[str],
    extra_cuda_cflags: Sequence[str] = ("-O3", "--use_fast_math"),
    verbose: bool = False,
    build_directory: Optional[Union[str, Path]] = None,
    env_prefix: str = "HYDROFORGE",
    force_rebuild: Optional[bool] = None,
    system_compiler: Optional[str] = None,
) -> Any:
    """Compile or fast-load an inline PyTorch CUDA extension.

    This is the shared hydroforge implementation used by every compiled-CUDA
    backend.  It keeps a config/source fingerprint beside the built ``.so`` and
    directly imports the cached binary when the fingerprint still matches,
    avoiding PyTorch's per-process ninja staleness check.

    Under a ROCm/HIP PyTorch build the same ``.cu`` sources double as the AMD
    compiled path: PyTorch's ``load_inline`` auto-hipifies the source, and only
    the nvcc-only ``--use_fast_math`` flag is renamed to ``-ffast-math`` so
    callers never need a separate HIP loader.

    Environment knobs:
      - ``HYDROFORGE_CUDA_REBUILD=1``: force rebuild.
      - ``HYDROFORGE_CUDA_USE_SYSTEM_COMPILER=auto|1|0``: host compiler fallback.
      - ``HYDROFORGE_CUDA_DIGEST_BUILD_DIR=0``: disable digest-scoped build dirs.
    """
    from torch.utils.cpp_extension import _get_build_directory, load_inline

    env_kind = "CUDA"
    if force_rebuild is None:
        force_rebuild = (
            _env_truthy(f"{env_prefix}_{env_kind}_REBUILD", False)
            or _env_truthy(f"HYDROFORGE_{env_kind}_REBUILD", False)
        )
    if system_compiler is None:
        system_compiler = os.environ.get(
            f"{env_prefix}_{env_kind}_USE_SYSTEM_COMPILER",
            os.environ.get(f"HYDROFORGE_{env_kind}_USE_SYSTEM_COMPILER", "auto"),
        )

    # ROCm/HIP build: PyTorch's load_inline hipifies the source itself; we only
    # rename the one device compile flag it leaves untouched.  Done before
    # fingerprinting so the cache key matches what actually gets compiled.
    if torch.version.hip is not None:
        extra_cuda_cflags = _normalise_hip_cflags(extra_cuda_cflags)

    fingerprint = _build_fingerprint(
        name=name,
        cpp_sources=cpp_sources,
        cuda_sources=cuda_sources,
        functions=functions,
        extra_cuda_cflags=extra_cuda_cflags,
    )
    digest = _fingerprint_digest(fingerprint)
    compiled_name = f"{name}_{digest[:16]}"
    cache_key = f"inline:{compiled_name}"
    if cache_key in _module_cache:
        return _module_cache[cache_key]

    if build_directory is not None:
        base_build_dir = Path(build_directory)
    else:
        base_build_dir = Path(_get_build_directory(name, verbose=False)) / _host_build_scope()
    digest_build_dir = (
        _env_truthy(f"{env_prefix}_{env_kind}_DIGEST_BUILD_DIR", True)
        and _env_truthy(f"HYDROFORGE_{env_kind}_DIGEST_BUILD_DIR", True)
    )
    build_dir = base_build_dir / digest[:16] if digest_build_dir else base_build_dir
    build_dir.mkdir(parents=True, exist_ok=True)
    so_path = build_dir / f"{compiled_name}.so"
    hash_path = build_dir / f"{compiled_name}.srchash"

    if not force_rebuild and so_path.is_file() and hash_path.is_file():
        try:
            if hash_path.read_text().strip() == digest:
                mod = _import_extension_from_so(compiled_name, so_path)
                _module_cache[cache_key] = mod
                return mod
        except Exception:
            pass

    _verbose = (
        verbose
        or os.environ.get(f"{env_prefix}_{env_kind}_VERBOSE", "") == "1"
        or os.environ.get(f"HYDROFORGE_{env_kind}_VERBOSE", "") == "1"
    )
    _maybe_clear_stale_torch_lock(build_dir, env_prefix=env_prefix, env_kind=env_kind, verbose=_verbose)
    old_compiler_env = _maybe_use_system_compiler(system_compiler)
    try:
        mod = load_inline(
            name=compiled_name,
            cpp_sources=cpp_sources,
            cuda_sources=cuda_sources,
            functions=list(functions),
            extra_cuda_cflags=list(extra_cuda_cflags),
            build_directory=str(build_dir),
            verbose=_verbose,
        )
    finally:
        _restore_compiler_env(old_compiler_env)

    try:
        hash_path.write_text(digest)
        update_build_manifest(
            build_dir,
            name,
            fingerprint | {"digest": digest, "compiled_module_name": compiled_name},
        )
    except Exception:
        pass

    _module_cache[cache_key] = mod
    return mod


def _local_rank() -> int:
    for name in (
        "LOCAL_RANK",
        "SLURM_LOCALID",
        "OMPI_COMM_WORLD_LOCAL_RANK",
        "MPI_LOCALRANKID",
        "MV2_COMM_WORLD_LOCAL_RANK",
    ):
        val = os.environ.get(name)
        if val is None:
            continue
        try:
            return int(val)
        except ValueError:
            return 0
    return 0


def _precompile_verbose(env_prefix: str) -> bool:
    return (
        os.environ.get(f"{env_prefix}_CUDA_VERBOSE") == "1"
        or os.environ.get("HYDROFORGE_CUDA_VERBOSE") == "1"
    )


def precompile_extension_builders(
    module_name: str,
    builders: Sequence[Tuple[str, str]],
    *,
    env_prefix: str = "HYDROFORGE",
    default_jobs: Optional[int] = None,
    local_rank0_only: bool = True,
) -> Dict[str, Any]:
    """Precompile independent extension builders, then fast-load them.

    ``builders`` is a sequence of ``(label, builder_function_name)`` pairs from
    ``module_name``.  Rank-local process 0 fans out to up to six independent
    Python subprocesses by default; other local ranks compile/load serially unless an
    explicit ``HYDROFORGE_PRECOMPILE_JOBS`` override is set.  This keeps multi-GPU jobs
    from multiplying compile fan-out by local world size while still relying on
    PyTorch's per-build-directory lock for cross-rank cache safety.

    Environment controls:
      - ``HYDROFORGE_PRECOMPILE_JOBS=N``: override extension fan-out.

    ``MAX_JOBS`` is still owned by PyTorch/Ninja and controls parallelism inside
    each individual extension build.
    """
    if not builders:
        return {}

    try:
        jobs = max(1, int(os.environ["HYDROFORGE_PRECOMPILE_JOBS"]))
    except (KeyError, ValueError):
        if local_rank0_only and _local_rank() != 0:
            jobs = 1
        else:
            jobs = default_jobs or 6
    jobs = min(len(builders), jobs)

    if jobs > 1:
        pythonpath = [p for p in sys.path if p]
        old_pythonpath = os.environ.get("PYTHONPATH")
        if old_pythonpath:
            pythonpath.append(old_pythonpath)

        verbose = _precompile_verbose("HYDROFORGE")
        pending = list(builders)
        running: List[Tuple[str, str, subprocess.Popen]] = []

        def start(label: str, builder: str) -> None:
            env = os.environ.copy()
            env["HYDROFORGE_PRECOMPILE_JOBS"] = "1"
            env["PYTHONPATH"] = os.pathsep.join(pythonpath)
            code = (
                "import importlib;"
                f"m=importlib.import_module({module_name!r});"
                f"getattr(m, {builder!r})()"
            )
            proc = subprocess.Popen(
                [sys.executable, "-c", code],
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=env,
            )
            running.append((label, builder, proc))

        try:
            while pending or running:
                while pending and len(running) < jobs:
                    start(*pending.pop(0))

                completed = False
                for item in list(running):
                    label, builder, proc = item
                    if proc.poll() is None:
                        continue
                    running.remove(item)
                    out, err = proc.communicate()
                    completed = True
                    if verbose:
                        if out:
                            print(out, end="", file=sys.stderr)
                        if err:
                            print(err, end="", file=sys.stderr)
                    if proc.returncode != 0:
                        msg = err or out
                        raise RuntimeError(
                            f"{module_name}.{builder} cuda precompile failed:\n{msg}"
                        )
                if not completed and running:
                    time.sleep(0.1)
        except Exception:
            for _, _, proc in running:
                if proc.poll() is None:
                    proc.terminate()
            for _, _, proc in running:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
            raise

    mod = importlib.import_module(module_name)
    return {label: getattr(mod, builder)() for label, builder in builders}


def precompile_cuda_modules(module_names: Iterable[str]) -> Dict[str, Any]:
    """Import modules and run their ``precompile_cuda_extensions`` hook."""
    results: Dict[str, Any] = {}
    for module_name in module_names:
        mod = importlib.import_module(module_name)
        hook = getattr(mod, "precompile_cuda_extensions", None)
        if hook is None:
            raise AttributeError(
                f"{module_name} does not define precompile_cuda_extensions()"
            )
        results[module_name] = hook()
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hydroforge-cuda-precompile",
        description="Precompile CUDA extensions declared by hydroforge model modules.",
    )
    parser.add_argument(
        "modules",
        nargs="+",
        help="Python modules exposing precompile_cuda_extensions(), e.g. cmfgpu.phys.cuda",
    )
    args = parser.parse_args(argv)
    precompile_cuda_modules(args.modules)
    return 0


# Sentinel for missing defaults
_MISSING = object()

# dtype string -> torch.dtype
_DTYPE_MAP = {
    "torch.bool": torch.bool,
    "torch.float32": torch.float32,
    "torch.float64": torch.float64,
    "torch.int32": torch.int32,
    "torch.int64": torch.int64,
    "bool": torch.bool,
    "float32": torch.float32,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
}


class CudaKernel:
    """Generic wrapper adapting a CUDA launcher to Triton calling convention.

    Replaces per-kernel wrapper classes like ``_OutflowKernel``.  The
    ``args`` list declares how Triton-style ``**kwargs`` map to the
    positional arguments of the C++ launcher function.

    Each entry in ``args`` can be:

    - ``"name_ptr"`` — a tensor argument; the kwarg key ``name_ptr`` is
      looked up directly in ``**kwargs``.
    - ``("name", type)`` — a scalar; ``type`` is ``float``, ``int``, or
      ``bool``.  The kwarg key ``name`` is looked up.
    - ``("name", type, default)`` — a scalar with a default value;
      ``kw.get(name, default)`` is used.

    The ``nullable`` dict maps tensor kwarg names to dtype strings.  If
    the kwarg is ``None`` or absent, an empty tensor of that dtype is
    substituted (required for pybind11 which cannot accept ``None``).

    Args:
        module: Compiled extension module (from :func:`load_cu_module`).
        func_name: Name of the launcher function (e.g. ``"launch_outflow"``).
        args: Argument specification list.
        nullable: ``{kwarg_name: dtype_str}`` for optional tensor args.
    """

    __slots__ = ("_func", "_args", "_nullable", "_fast_caller")

    def __init__(
        self,
        module: Any,
        func_name: str,
        args: List[Union[str, Tuple]],
        nullable: Optional[Dict[str, str]] = None,
    ):
        self._func = getattr(module, func_name)
        self._args = args
        self._nullable = nullable or {}
        self._fast_caller: Any = None

    def _build_fast_caller(self) -> Any:
        """Generate a specialised caller that avoids per-call arg iteration.

        Instead of looping over 20-31 arg specs on every call, we emit a
        function that directly maps kwargs to positional args by name.
        """
        ns: Dict[str, Any] = {"_func": self._func}
        body_lines: List[str] = []
        arg_names: List[str] = []

        for i, spec in enumerate(self._args):
            a = f"_a{i}"
            arg_names.append(a)
            if isinstance(spec, str):
                if spec in self._nullable:
                    dt = _DTYPE_MAP.get(self._nullable[spec], torch.bool)
                    ns[f"_empty{i}"] = torch.empty(0, dtype=dt)
                    body_lines.append(
                        f"  {a} = _kw.get('{spec}'); "
                        f"{a} = _empty{i} if {a} is None else {a}"
                    )
                else:
                    body_lines.append(f"  {a} = _kw['{spec}']")
            elif isinstance(spec, tuple):
                name = spec[0]
                if len(spec) >= 3:
                    ns[f"_def{i}"] = spec[2]
                    body_lines.append(f"  {a} = _kw.get('{name}', _def{i})")
                else:
                    body_lines.append(f"  {a} = _kw['{name}']")
            else:
                ns[f"_const{i}"] = spec
                body_lines.append(f"  {a} = _const{i}")

        call_args = ", ".join(arg_names)
        fn_body = "\n".join(body_lines)
        code = f"def _fast(_kw):\n{fn_body}\n  return _func({call_args})"
        exec(code, ns)  # noqa: S102 – generated code is fully controlled
        return ns["_fast"]

    def __call__(self, **kw: Any) -> Any:
        caller = self._fast_caller
        if caller is None:
            caller = self._build_fast_caller()
            self._fast_caller = caller
        return caller(kw)

    def __getitem__(self, grid):
        """Accept ``kernel[grid]`` syntax; grid is unused for CUDA."""
        return self


class LazyCudaKernel:
    """Lazy-initialized CudaKernel that defers compilation until first call.

    Accepts a zero-argument factory that returns a ``CudaKernel``.
    Supports the Triton ``kernel[grid](**kw)`` calling convention.

    Example::

        def _make():
            mod = get_cuda_kernels()
            return CudaKernel(mod, "launch_outflow", args=[...])

        compute_outflow_kernel = LazyCudaKernel(_make)
    """

    __slots__ = ("_factory", "_kernel")

    def __init__(self, factory):
        self._factory = factory
        self._kernel = None

    def __call__(self, **kw):
        if self._kernel is None:
            self._kernel = self._factory()
        return self._kernel(**kw)

    def __getitem__(self, grid):
        return self


if __name__ == "__main__":
    raise SystemExit(main())

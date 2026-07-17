# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Shared CUDA-extension loader for hydroforge compiled backends.

:func:`load_inline_cu_module` compiles inline ``.cu`` sources into a pybind11
extension and fast-loads the cached ``.so`` on subsequent runs.  The cache is
keyed by a content+toolchain fingerprint and is host-agnostic, so a binary
compiled by any host (or rank) is reused.  When a build is needed,
:func:`_coordinated_build` serialises compilation across processes / nodes with
an atomic filesystem lock so exactly one process compiles and the rest reuse its
output.  :func:`precompile_extension_builders` builds a model's independent
extensions in parallel before the hot loop.

Under a ROCm/HIP PyTorch build the same ``.cu`` sources double as the AMD path:
``load_inline`` auto-hipifies the source and only ``--use_fast_math`` is renamed
to ``-ffast-math``.
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
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import torch

_module_cache: Dict[str, Any] = {}

_MANIFEST_FILENAME = "compile_manifest.json"


def _safe_path_component(value: str) -> str:
    component = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip())
    return component[:80] or "unknown"


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
    extra_ldflags: Sequence[str] = (),
    extra_include_paths: Sequence[str] = (),
) -> dict:
    from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

    return {
        "module_name": name,
        "toolchain": "hip" if torch.version.hip is not None else "cuda",
        "cpp_sha256": hashlib.sha256(_normalise_inline_source(cpp_sources).encode()).hexdigest(),
        "cuda_sha256": hashlib.sha256(_normalise_inline_source(cuda_sources).encode()).hexdigest(),
        "functions": list(functions),
        "extra_cuda_cflags": list(extra_cuda_cflags),
        "extra_ldflags": list(extra_ldflags),
        "extra_include_paths": list(extra_include_paths),
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


def _host_scoped_cache(env_prefix: str) -> bool:
    """Whether to isolate the build cache per hostname (default: no).

    Host-agnostic caching lets a ``.so`` compiled on one host be reused by any
    other host whose build *fingerprint* matches — the digest already encodes
    binary compatibility (arch, toolchain, torch/cuda version, compiler), so the
    hostname is irrelevant.  Set ``HYDROFORGE_CUDA_HOST_SCOPED_CACHE=1`` to keep
    the legacy per-host subdirectory (e.g. on a shared filesystem where you do
    *not* want cross-host reuse).
    """
    return (
        _env_truthy(f"{env_prefix}_CUDA_HOST_SCOPED_CACHE", False)
        or _env_truthy("HYDROFORGE_CUDA_HOST_SCOPED_CACHE", False)
    )


def _candidate_build_dirs(base_no_host: Path, digest16: str) -> List[Path]:
    """Directories to search for an already-compiled ``.so`` (host-agnostic).

    Returns the canonical host-agnostic dir first, then any legacy per-host
    siblings (``base/<host>/<digest>``) so caches built before host-agnostic
    mode — or by another host writing into its own scope — are still reused.
    """
    cands = [base_no_host / digest16]
    try:
        for child in sorted(base_no_host.iterdir()):
            if child.is_dir() and child.name != digest16:
                sib = child / digest16
                if sib.is_dir():
                    cands.append(sib)
    except OSError:
        pass
    return cands


def _try_import_cached(
    compiled_name: str, digest: str, search_dirs: Sequence[Path]
) -> Optional[Any]:
    """Import the first valid cached ``.so`` whose srchash matches ``digest``."""
    for d in search_dirs:
        so_path = d / f"{compiled_name}.so"
        hash_path = d / f"{compiled_name}.srchash"
        if so_path.is_file() and hash_path.is_file():
            try:
                if hash_path.read_text().strip() == digest:
                    return _import_extension_from_so(compiled_name, so_path)
            except Exception:
                continue
    return None


def _read_compile_lock_holder(lock_path: Path) -> Tuple[Optional[str], Optional[int]]:
    try:
        raw = lock_path.read_text(errors="replace").strip()
    except OSError:
        return None, None
    parts = raw.split(":", 2)
    if len(parts) < 2:
        return None, None
    try:
        pid = int(parts[1])
    except ValueError:
        pid = None
    return parts[0] or None, pid


def _process_is_alive(pid: Optional[int]) -> bool:
    if pid is None or pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _maybe_remove_abandoned_compile_lock(
    lock_path: Path,
    *,
    stale_after: float,
    env_prefix: str,
    verbose: bool,
) -> bool:
    if stale_after <= 0.0:
        return False
    try:
        age = time.time() - lock_path.stat().st_mtime
    except OSError:
        return False
    if age < stale_after:
        return False

    holder_host, holder_pid = _read_compile_lock_holder(lock_path)
    local_host = socket.gethostname()
    if holder_host == local_host and not _process_is_alive(holder_pid):
        reason = f"abandoned by local pid {holder_pid}"
    elif (
        _env_truthy(f"{env_prefix}_CUDA_COMPILE_LOCK_STEAL", False)
        or _env_truthy("HYDROFORGE_CUDA_COMPILE_LOCK_STEAL", False)
    ):
        reason = "explicit stale-lock steal enabled"
    else:
        return False

    try:
        lock_path.unlink()
    except FileNotFoundError:
        return True
    if verbose:
        print(
            f"[hydroforge] removed compile lock {lock_path} "
            f"({reason}, age {age:.0f}s)",
            file=sys.stderr,
        )
    return True


def _acquire_compile_lock(lock_path: Path, *, env_prefix: str, verbose: bool) -> None:
    stale_after = _env_float(
        f"{env_prefix}_CUDA_COMPILE_LOCK_STALE_SECONDS",
        _env_float("HYDROFORGE_CUDA_COMPILE_LOCK_STALE_SECONDS", 1800.0),
    )
    poll = max(0.05, _env_float("HYDROFORGE_CUDA_COMPILE_LOCK_POLL_SECONDS", 0.25))
    timeout = _env_float(
        f"{env_prefix}_CUDA_COMPILE_LOCK_TIMEOUT_SECONDS",
        _env_float("HYDROFORGE_CUDA_COMPILE_LOCK_TIMEOUT_SECONDS", 1800.0),
    )
    deadline = time.time() + timeout if timeout > 0.0 else None

    while True:
        try:
            fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            if _maybe_remove_abandoned_compile_lock(
                lock_path,
                stale_after=stale_after,
                env_prefix=env_prefix,
                verbose=verbose,
            ):
                continue
            if deadline is not None and time.time() > deadline:
                holder_host, holder_pid = _read_compile_lock_holder(lock_path)
                holder = (
                    f"{holder_host or 'unknown'}:{holder_pid}"
                    if holder_pid is not None
                    else (holder_host or "unknown")
                )
                raise TimeoutError(
                    f"Timed out waiting for CUDA compile lock {lock_path} "
                    f"held by {holder}. Remove the lock if the compiler process "
                    "has exited, or set HYDROFORGE_CUDA_COMPILE_LOCK_STEAL=1 "
                    "to override a stale lock."
                )
            time.sleep(poll)
            continue
        else:
            try:
                os.write(
                    fd,
                    f"{socket.gethostname()}:{os.getpid()}:{time.time():.0f}".encode(),
                )
            finally:
                os.close(fd)
            return


def _release_compile_lock(lock_path: Path) -> None:
    try:
        lock_path.unlink()
    except FileNotFoundError:
        pass


def _coordinated_build(
    *,
    compiled_name: str,
    digest: str,
    base_no_host: Path,
    host_dir: Path,
    host_scoped: bool,
    force_rebuild: bool,
    compile_fn,
    env_prefix: str,
    verbose: bool,
) -> Any:
    """Find-or-build a CUDA extension with cross-process / cross-host coordination.

    1. **Fast path** — import an existing matching ``.so`` from any candidate dir
       (host-agnostic reuse: a binary compiled by *any* host/rank with the same
       digest is reused).
    2. **Slow path** — acquire an atomic ``O_CREAT|O_EXCL`` build lock in the
       canonical digest dir so *exactly one* process compiles; the rest poll,
       re-checking the cache until the compiler finishes (then fast-load its
       output).  ``O_CREAT|O_EXCL`` is atomic on NFS, so this serialises across
       nodes on a shared filesystem without any ``torch.distributed`` handshake.
       Abandoned local locks are removed only when the recorded PID is gone;
       remote stale locks require an explicit override.

    ``compile_fn(build_dir)`` runs the actual ``load_inline`` into ``build_dir``
    and returns the module.
    """
    digest16 = digest[:16]
    canonical = (host_dir if host_scoped else base_no_host) / digest16
    search = [canonical] if host_scoped else _candidate_build_dirs(base_no_host, digest16)

    if not force_rebuild:
        mod = _try_import_cached(compiled_name, digest, search)
        if mod is not None:
            return mod

    canonical.mkdir(parents=True, exist_ok=True)
    lock_path = canonical / ".hydroforge_compile.lock"
    _acquire_compile_lock(lock_path, env_prefix=env_prefix, verbose=verbose)
    try:
        # Double-check: a previous holder may have finished between our cache
        # miss and acquiring the lock.
        if not force_rebuild:
            mod = _try_import_cached(compiled_name, digest, search)
            if mod is not None:
                return mod
        return compile_fn(canonical)
    finally:
        _release_compile_lock(lock_path)


def load_inline_cu_module(
    name: str,
    *,
    cpp_sources: Union[str, Sequence[str]],
    cuda_sources: Union[str, Sequence[str]],
    functions: Sequence[str],
    extra_cuda_cflags: Sequence[str] = ("-O3", "--use_fast_math"),
    extra_ldflags: Sequence[str] = (),
    extra_include_paths: Sequence[str] = (),
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
        extra_ldflags=extra_ldflags,
        extra_include_paths=extra_include_paths,
    )
    digest = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    compiled_name = f"{name}_{digest[:16]}"
    cache_key = f"inline:{compiled_name}"
    if cache_key in _module_cache:
        return _module_cache[cache_key]

    if build_directory is not None:
        base_no_host = Path(build_directory)
        host_dir = base_no_host
    else:
        cache_root = Path(_get_build_directory(name, verbose=False))
        base_no_host = cache_root
        host_dir = cache_root / _safe_path_component(socket.gethostname() or "host")
    host_scoped = _host_scoped_cache(env_prefix)
    digest_build_dir = (
        _env_truthy(f"{env_prefix}_{env_kind}_DIGEST_BUILD_DIR", True)
        and _env_truthy(f"HYDROFORGE_{env_kind}_DIGEST_BUILD_DIR", True)
    )
    if not digest_build_dir:
        # Legacy flat layout (no per-digest subdir): keep the original
        # single-directory layout but still serialize through the hydroforge lock.
        build_dir = host_dir if host_scoped else base_no_host
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

    def _compile_into(build_dir: Path) -> Any:
        """Run the real ``load_inline`` into ``build_dir`` and stamp the cache."""
        build_dir.mkdir(parents=True, exist_ok=True)
        _maybe_clear_stale_torch_lock(
            build_dir, env_prefix=env_prefix, env_kind=env_kind, verbose=_verbose
        )
        old_compiler_env = _maybe_use_system_compiler(system_compiler)
        try:
            mod = load_inline(
                name=compiled_name,
                cpp_sources=cpp_sources,
                cuda_sources=cuda_sources,
                functions=list(functions),
                extra_cuda_cflags=list(extra_cuda_cflags),
                extra_ldflags=list(extra_ldflags),
                extra_include_paths=list(extra_include_paths),
                build_directory=str(build_dir),
                verbose=_verbose,
            )
        finally:
            _restore_compiler_env(old_compiler_env)
        try:
            (build_dir / f"{compiled_name}.srchash").write_text(digest)
            update_build_manifest(
                build_dir,
                name,
                fingerprint | {"digest": digest, "compiled_module_name": compiled_name},
            )
        except Exception:
            pass
        return mod

    if not digest_build_dir:
        build_dir = host_dir if host_scoped else base_no_host
        lock_path = build_dir / ".hydroforge_compile.lock"
        _acquire_compile_lock(lock_path, env_prefix=env_prefix, verbose=_verbose)
        try:
            if not force_rebuild:
                so_path = build_dir / f"{compiled_name}.so"
                hash_path = build_dir / f"{compiled_name}.srchash"
                if so_path.is_file() and hash_path.is_file():
                    try:
                        if hash_path.read_text().strip() == digest:
                            mod = _import_extension_from_so(compiled_name, so_path)
                            _module_cache[cache_key] = mod
                            return mod
                    except Exception:
                        pass
            mod = _compile_into(build_dir)
        finally:
            _release_compile_lock(lock_path)
    else:
        mod = _coordinated_build(
            compiled_name=compiled_name,
            digest=digest,
            base_no_host=base_no_host,
            host_dir=host_dir,
            host_scoped=host_scoped,
            force_rebuild=force_rebuild,
            compile_fn=_compile_into,
            env_prefix=env_prefix,
            verbose=_verbose,
        )

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
    the shared hydroforge compile lock for cross-rank cache safety.

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

        verbose = os.environ.get("HYDROFORGE_CUDA_VERBOSE") == "1"
        pending = list(builders)
        running: List[Tuple[str, str, subprocess.Popen, Path]] = []

        def read_log(log_path: Path, *, tail_bytes: Optional[int] = None) -> str:
            try:
                if tail_bytes is None:
                    return log_path.read_text(errors="replace")
                with open(log_path, "rb") as f:
                    try:
                        f.seek(0, os.SEEK_END)
                        size = f.tell()
                        f.seek(max(0, size - tail_bytes))
                    except OSError:
                        f.seek(0)
                    return f.read().decode(errors="replace")
            except OSError:
                return ""

        def remove_log(log_path: Path) -> None:
            try:
                log_path.unlink()
            except OSError:
                pass

        def start(label: str, builder: str) -> None:
            env = os.environ.copy()
            env["HYDROFORGE_PRECOMPILE_JOBS"] = "1"
            env["PYTHONPATH"] = os.pathsep.join(pythonpath)
            code = (
                "import importlib;"
                f"m=importlib.import_module({module_name!r});"
                f"getattr(m, {builder!r})()"
            )
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f"hydroforge_cuda_{_safe_path_component(label)}_",
                suffix=".log",
                delete=False,
            ) as log:
                log_path = Path(log.name)
                try:
                    proc = subprocess.Popen(
                        [sys.executable, "-c", code],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        env=env,
                    )
                except Exception:
                    remove_log(log_path)
                    raise
            running.append((label, builder, proc, log_path))

        try:
            while pending or running:
                while pending and len(running) < jobs:
                    start(*pending.pop(0))

                completed = False
                for item in list(running):
                    label, builder, proc, log_path = item
                    if proc.poll() is None:
                        continue
                    running.remove(item)
                    completed = True
                    log_text = read_log(log_path, tail_bytes=None if verbose else 256 * 1024)
                    if verbose:
                        if log_text:
                            print(log_text, end="", file=sys.stderr)
                    remove_log(log_path)
                    if proc.returncode != 0:
                        msg = log_text or "<no compiler output captured>"
                        raise RuntimeError(
                            f"{module_name}.{builder} cuda precompile failed:\n{msg}"
                        )
                if not completed and running:
                    time.sleep(0.1)
        except Exception:
            for _, _, proc, _ in running:
                if proc.poll() is None:
                    proc.terminate()
            for _, _, proc, log_path in running:
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                remove_log(log_path)
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


if __name__ == "__main__":
    raise SystemExit(main())

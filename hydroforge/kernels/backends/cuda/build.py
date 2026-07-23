"""Content-addressed compilation and coordinated CUDA extension cache."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import shutil
import socket
import sys
import sysconfig
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from hydroforge.serialization.files import atomic_write_text

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
    atomic_write_text(p, json.dumps(manifest, indent=2))


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


def _build_fingerprint(
    *,
    name: str,
    cpp_sources: Union[str, Sequence[str]],
    cuda_sources: Union[str, Sequence[str]],
    functions: Sequence[str],
    extra_cuda_cflags: Sequence[str],
    extra_ldflags: Sequence[str] = (),
    extra_include_paths: Sequence[str] = (),
    compiler_selection: tuple[str, str | None, str | None] = (
        "never", None, None,
    ),
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
        # PyTorch extensions link against both CPython and the host C++ ABI.
        # A shared cache must never reuse a binary produced for another Python
        # extension ABI or machine platform merely because Torch/source match.
        "python_cache_tag": sys.implementation.cache_tag,
        "python_soabi": sysconfig.get_config_var("SOABI"),
        "python_platform": sysconfig.get_platform(),
        "cuda_home": str(CUDA_HOME) if CUDA_HOME is not None else None,
        "rocm_home": str(ROCM_HOME) if ROCM_HOME is not None else None,
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
        "pytorch_rocm_arch": os.environ.get("PYTORCH_ROCM_ARCH"),
        "visible_device_arches": _visible_device_arches(),
        "cc": os.environ.get("CC"),
        "cxx": os.environ.get("CXX"),
        "cuda_host_cxx": os.environ.get("CUDAHOSTCXX"),
        "system_compiler_mode": compiler_selection[0],
        "resolved_system_cc": compiler_selection[1],
        "resolved_system_cxx": compiler_selection[2],
    }


def _resolve_system_compiler(
    mode: str = "never",
) -> tuple[str, str | None, str | None]:
    """Resolve the exact compiler choice before cache fingerprinting."""

    mode = mode.strip().lower()
    if mode in {"0", "false", "no", "off", "never"}:
        return "never", None, None
    if mode == "auto":
        raise ValueError(
            "CUDA system compiler mode 'auto' is forbidden; choose 'always' "
            "or 'never' explicitly"
        )
    if mode not in {"1", "true", "yes", "on", "always"}:
        raise ValueError(f"invalid CUDA system compiler mode {mode!r}")

    gcc = shutil.which("gcc")
    gxx = shutil.which("g++")
    if not gcc or not gxx:
        raise RuntimeError("system gcc/g++ were requested but are unavailable")
    return "always", gcc, gxx


def _activate_system_compiler(
    selection: tuple[str, str | None, str | None],
) -> Dict[str, Optional[str]]:
    mode, gcc, gxx = selection
    if mode == "never":
        return {}
    if mode != "always" or gcc is None or gxx is None:
        raise RuntimeError(f"invalid resolved compiler selection {selection!r}")

    old = {"CC": os.environ.get("CC"), "CXX": os.environ.get("CXX")}
    os.environ["CC"] = gcc
    os.environ["CXX"] = gxx
    return old


def _configure_system_compiler(mode: str = "never") -> Dict[str, Optional[str]]:
    """Temporarily activate one explicit, prevalidated compiler choice."""

    return _activate_system_compiler(_resolve_system_compiler(mode))


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
    except ValueError as exc:
        raise ValueError(f"{name} must be a floating-point number, got {val!r}") from exc


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


def _try_import_cached(
    compiled_name: str, digest: str, search_dirs: Sequence[Path]
) -> Optional[Any]:
    """Import the first valid cached ``.so`` whose srchash matches ``digest``."""
    for d in search_dirs:
        so_path = d / f"{compiled_name}.so"
        hash_path = d / f"{compiled_name}.srchash"
        if so_path.is_file() and hash_path.is_file():
            if hash_path.read_text().strip() == digest:
                return _import_extension_from_so(compiled_name, so_path)
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
    cache_root: Path,
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
    canonical = cache_root / digest16
    search = [canonical]

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
      - ``HYDROFORGE_CUDA_USE_SYSTEM_COMPILER=always|never``: explicit host compiler.
    Every binary is stored in the one canonical fingerprint directory. There
    is no host-scoped or flat-layout compatibility lookup.
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
            os.environ.get(f"HYDROFORGE_{env_kind}_USE_SYSTEM_COMPILER", "always"),
        )
    compiler_selection = _resolve_system_compiler(system_compiler)

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
        compiler_selection=compiler_selection,
    )
    digest = hashlib.sha256(
        json.dumps(fingerprint, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    compiled_name = f"{name}_{digest[:16]}"
    cache_key = f"inline:{compiled_name}"
    if not force_rebuild and cache_key in _module_cache:
        return _module_cache[cache_key]

    if build_directory is not None:
        cache_root = Path(build_directory)
    else:
        cache_root = Path(_get_build_directory(name, verbose=False))

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
        old_compiler_env = _activate_system_compiler(compiler_selection)
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
        atomic_write_text(
            build_dir / f"{compiled_name}.srchash", digest,
        )
        update_build_manifest(
            build_dir,
            name,
            fingerprint | {"digest": digest, "compiled_module_name": compiled_name},
        )
        return mod

    mod = _coordinated_build(
        compiled_name=compiled_name,
        digest=digest,
        cache_root=cache_root,
        force_rebuild=force_rebuild,
        compile_fn=_compile_into,
        env_prefix=env_prefix,
        verbose=_verbose,
    )

    _module_cache[cache_key] = mod
    return mod

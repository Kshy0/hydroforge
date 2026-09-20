"""Content-addressed compilation and coordinated CUDA extension cache."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import math
import os
import shutil
import socket
import sys
import sysconfig
import time
from collections.abc import Callable, Sequence
from contextlib import contextmanager
from decimal import Decimal
from pathlib import Path
from typing import Any

import torch
from pydantic import validate_call

from hydroforge.contracts.errors import cleanup_on_exit
from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.kernels.backends.build_environment import (
    serialized_compilation as _serialized_compilation,
)
from hydroforge.kernels.backends.build_environment import temporary_environment
from hydroforge.serialization.files import atomic_write_text

_module_cache: dict[str, Any] = {}
_MANIFEST_FILENAME = "compile_manifest.json"


def _safe_path_component(value: str) -> str:
    component = "".join(
        ch if ch.isalnum() or ch in "._-" else "_" for ch in value.strip()
    )
    return component[:80] or "unknown"


def load_build_manifest(build_dir: str | Path) -> dict:
    """Load the compile manifest from a build directory."""
    p = Path(build_dir) / _MANIFEST_FILENAME
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


def update_build_manifest(build_dir: str | Path, section: str, data: dict) -> None:
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


def _normalise_inline_source(src: str | Sequence[str]) -> str:
    if isinstance(src, str):
        return src
    return "\n".join(src)


def _content_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _include_path_fingerprints(paths: Sequence[str]) -> list[dict[str, Any]]:
    """Fingerprint every file visible through an explicit include path."""

    def linked_target(path: Path) -> str | None:
        if not path.is_symlink():
            return None
        try:
            return str(path.resolve(strict=True))
        except FileNotFoundError:
            return str(path.resolve())
        except (OSError, RuntimeError) as error:
            raise ValueError(
                f"cannot resolve CUDA include link {path}: {error}"
            ) from error

    fingerprints = []
    for raw_path in paths:
        path = Path(raw_path)
        entry: dict[str, Any] = {
            "path": str(path),
            "version": 2,
            "link_target": linked_target(path),
        }
        if path.is_file():
            entry.update(kind="file", sha256=_content_sha256(path))
        elif path.is_dir():
            digest = hashlib.sha256()

            def visit(directory: Path, ancestors: frozenset[tuple[int, int]]) -> int:
                status = directory.stat()
                identity = (status.st_dev, status.st_ino)
                if identity in ancestors:
                    raise ValueError(f"cyclic CUDA include directory link: {directory}")
                ancestors = ancestors | {identity}
                count = 0
                for candidate in sorted(
                    directory.iterdir(), key=lambda item: item.name
                ):
                    target = linked_target(candidate)
                    record = json.dumps(
                        [candidate.relative_to(path).as_posix(), target],
                        ensure_ascii=True,
                    ).encode()
                    digest.update(len(record).to_bytes(8, "big"))
                    digest.update(record)
                    if candidate.is_dir():
                        digest.update(b"directory\0")
                        count += visit(candidate, ancestors)
                    elif candidate.is_file():
                        digest.update(b"file\0")
                        digest.update(bytes.fromhex(_content_sha256(candidate)))
                        count += 1
                    else:
                        digest.update(b"missing\0")
                return count

            count = visit(path, frozenset())
            entry.update(
                kind="directory",
                sha256=digest.hexdigest(),
                file_count=count,
            )
        else:
            entry.update(kind="missing", sha256=None)
        fingerprints.append(entry)
    return fingerprints


def _normalise_hip_cflags(flags: Sequence[str]) -> list[str]:
    """Translate supported NVCC options without discarding numerical controls.

    PyTorch hipifies source but not compiler flags. Keep option order so an
    explicit FTZ setting can override fast math. Unknown options remain visible
    to the compiler rather than being silently dropped.
    """
    replacements = {
        "--use_fast_math": "-ffast-math",
        "-use_fast_math": "-ffast-math",
        "--ftz=false": "-fno-gpu-flush-denormals-to-zero",
        "-ftz=false": "-fno-gpu-flush-denormals-to-zero",
        "--ftz=true": "-fgpu-flush-denormals-to-zero",
        "-ftz=true": "-fgpu-flush-denormals-to-zero",
    }
    return [replacements.get(flag, flag) for flag in flags]


def _visible_device_arches() -> list[str]:
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
    cpp_sources: str | Sequence[str],
    cuda_sources: str | Sequence[str],
    functions: Sequence[str],
    extra_cuda_cflags: Sequence[str],
    extra_ldflags: Sequence[str] = (),
    extra_include_paths: Sequence[str] = (),
    compiler_selection: tuple[str, str | None, str | None] = (
        "never",
        None,
        None,
    ),
) -> dict:
    from torch.utils.cpp_extension import CUDA_HOME, ROCM_HOME

    fingerprint = {
        "module_name": name,
        "toolchain": "hip" if torch.version.hip is not None else "cuda",
        "cpp_sha256": hashlib.sha256(
            _normalise_inline_source(cpp_sources).encode()
        ).hexdigest(),
        "cuda_sha256": hashlib.sha256(
            _normalise_inline_source(cuda_sources).encode()
        ).hexdigest(),
        "functions": list(functions),
        "extra_cuda_cflags": list(extra_cuda_cflags),
        "extra_ldflags": list(extra_ldflags),
        "extra_include_paths": list(extra_include_paths),
        "extra_include_contents": _include_path_fingerprints(
            extra_include_paths,
        ),
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
    compiler_environment = {
        name: value
        for name in (
            "NVCC_PREPEND_FLAGS",
            "NVCC_APPEND_FLAGS",
            "NVCC_CCBIN",
            "PYTORCH_NVCC",
        )
        if (value := os.environ.get(name)) is not None
    }
    if compiler_environment:
        fingerprint["cuda_environment"] = compiler_environment
    return fingerprint


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


@contextmanager
def _temporary_build_environment(selection: tuple[str, str | None, str | None]):
    """Restore owned values even when activation or another restoration fails."""
    mode, gcc, gxx = selection
    if mode not in {"always", "never"} or (
        mode == "always" and (gcc is None or gxx is None)
    ):
        raise RuntimeError(f"invalid resolved compiler selection {selection!r}")
    old_path = os.environ.get("PATH")
    executable_dir = sysconfig.get_path("scripts") or str(Path(sys.executable).parent)
    entries = [] if old_path is None else old_path.split(os.pathsep)
    if executable_dir not in entries:
        entries.insert(0, executable_dir)
    changes = {"PATH": os.pathsep.join(entries)}
    if mode == "always":
        changes.update(CC=gcc, CXX=gxx)
    with temporary_environment(changes):
        yield


def _import_extension_from_so(name: str, so_path: Path) -> Any:
    import torch  # noqa: F401 - ensure torch shared libraries are loaded first

    spec = importlib.util.spec_from_file_location(name, str(so_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot create extension spec for {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _env_float(name: str, default: float, *, fallback: str | None = None) -> float:
    """Resolve a finite float without evaluating an unused fallback variable."""

    val = os.environ.get(name)
    if val is None and fallback is not None:
        name = fallback
        val = os.environ.get(name)
    if val is None:
        return default
    try:
        result = float(val)
    except ValueError as exc:
        raise ValueError(
            f"{name} must be a floating-point number, got {val!r}"
        ) from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {val!r}")
    if result == 0.0 and not Decimal(val.lower().partition("e")[0]).is_zero():
        raise ValueError(f"{name} contains a nonzero value that underflows float64")
    return result


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
        900.0,
        fallback=f"HYDROFORGE_{env_kind}_LOCK_STALE_SECONDS",
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
) -> Any | None:
    """Import the first valid cached ``.so`` whose srchash matches ``digest``."""
    for d in search_dirs:
        so_path = d / f"{compiled_name}.so"
        hash_path = d / f"{compiled_name}.srchash"
        if so_path.is_file() and hash_path.is_file():
            if hash_path.read_text().strip() == digest:
                return _import_extension_from_so(compiled_name, so_path)
    return None


def _read_compile_lock_holder(lock_path: Path) -> tuple[str | None, int | None]:
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


def _process_is_alive(pid: int | None) -> bool:
    if pid is None or pid <= 0:
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def _compile_lock_guard(lock_path: Path):
    """Serialize lock-file create/remove decisions on one stable inode."""
    guard_path = lock_path.with_name(f"{lock_path.name}.guard")
    descriptor = os.open(str(guard_path), os.O_CREAT | os.O_RDWR, 0o644)
    cleanup = [lambda: os.close(descriptor)]
    with cleanup_on_exit("CUDA compile lock guard", cleanup):
        if os.name == "posix":
            import fcntl

            fcntl.flock(descriptor, fcntl.LOCK_EX)
            cleanup.insert(0, lambda: fcntl.flock(descriptor, fcntl.LOCK_UN))
        yield


def _remove_abandoned_compile_lock_unlocked(
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
    holder_host, holder_pid = _read_compile_lock_holder(lock_path)
    local_host = socket.gethostname()
    dead_local_holder = holder_host == local_host and not _process_is_alive(holder_pid)
    if dead_local_holder:
        grace = max(
            0.0,
            _env_float(
                f"{env_prefix}_CUDA_COMPILE_LOCK_DEAD_PID_GRACE_SECONDS",
                2.0,
                fallback="HYDROFORGE_CUDA_COMPILE_LOCK_DEAD_PID_GRACE_SECONDS",
            ),
        )
        if age < min(stale_after, grace):
            return False
        reason = f"abandoned by local pid {holder_pid}"
    elif age >= stale_after and (
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
            f"[hydroforge] removed compile lock {lock_path} ({reason}, age {age:.0f}s)",
            file=sys.stderr,
        )
    return True


def _acquire_compile_lock(
    lock_path: Path,
    *,
    env_prefix: str,
    verbose: bool,
    cache_probe: Callable[[], Any | None] | None = None,
) -> Any | None:
    stale_after = _env_float(
        f"{env_prefix}_CUDA_COMPILE_LOCK_STALE_SECONDS",
        1800.0,
        fallback="HYDROFORGE_CUDA_COMPILE_LOCK_STALE_SECONDS",
    )
    poll = max(0.05, _env_float("HYDROFORGE_CUDA_COMPILE_LOCK_POLL_SECONDS", 0.25))
    timeout = _env_float(
        f"{env_prefix}_CUDA_COMPILE_LOCK_TIMEOUT_SECONDS",
        1800.0,
        fallback="HYDROFORGE_CUDA_COMPILE_LOCK_TIMEOUT_SECONDS",
    )
    deadline = time.time() + timeout if timeout > 0.0 else None

    while True:
        if cache_probe is not None:
            cached = cache_probe()
            if cached is not None:
                return cached
        removed = False
        with _compile_lock_guard(lock_path):
            try:
                fd = os.open(
                    str(lock_path),
                    os.O_CREAT | os.O_EXCL | os.O_WRONLY,
                    0o644,
                )
            except FileExistsError:
                removed = _remove_abandoned_compile_lock_unlocked(
                    lock_path,
                    stale_after=stale_after,
                    env_prefix=env_prefix,
                    verbose=verbose,
                )
            else:
                try:
                    with cleanup_on_exit(
                        "CUDA compile lock descriptor", (lambda: os.close(fd),)
                    ):
                        remaining = memoryview(
                            f"{socket.gethostname()}:{os.getpid()}:{time.time():.0f}".encode()
                        )
                        while remaining:
                            written = os.write(fd, remaining)
                            if written == 0:
                                raise OSError(
                                    "CUDA compile lock write made no progress"
                                )
                            remaining = remaining[written:]
                except BaseException:
                    # Only the successful O_EXCL creator owns this path, and
                    # the guard still excludes other create/remove decisions.
                    with cleanup_on_exit(
                        "incomplete CUDA compile lock", (lock_path.unlink,)
                    ):
                        raise
                return None
        if removed:
            continue
        if lock_path.exists():
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


def _release_compile_lock(lock_path: Path) -> None:
    with _compile_lock_guard(lock_path):
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
    cached = _acquire_compile_lock(
        lock_path,
        env_prefix=env_prefix,
        verbose=verbose,
        cache_probe=(
            None
            if force_rebuild
            else lambda: _try_import_cached(compiled_name, digest, search)
        ),
    )
    if cached is not None:
        return cached
    with cleanup_on_exit(
        "CUDA compile lock", (lambda: _release_compile_lock(lock_path),)
    ):
        # Double-check: a previous holder may have finished between our cache
        # miss and acquiring the lock.
        if not force_rebuild:
            mod = _try_import_cached(compiled_name, digest, search)
            if mod is not None:
                return mod
        return compile_fn(canonical)


@validate_call(config=HydroForgeModel.model_config)
@_serialized_compilation()
def load_inline_cu_module(
    name: str,
    *,
    cpp_sources: str | Sequence[str],
    cuda_sources: str | Sequence[str],
    functions: Sequence[str],
    extra_cuda_cflags: Sequence[str] = ("-O3", "--use_fast_math"),
    extra_ldflags: Sequence[str] = (),
    extra_include_paths: Sequence[str] = (),
    verbose: bool = False,
    build_directory: str | os.PathLike[str] | None = None,
    env_prefix: str = "HYDROFORGE",
    force_rebuild: bool | None = None,
    system_compiler: str | None = None,
    cache_only: bool = False,
) -> Any:
    """Compile or fast-load an inline PyTorch CUDA extension.

    This is the shared hydroforge implementation used by every compiled-CUDA
    backend.  It keeps a config/source fingerprint beside the built ``.so`` and
    directly imports the cached binary when the fingerprint still matches,
    avoiding PyTorch's per-process ninja staleness check.

    Under a ROCm/HIP PyTorch build the same ``.cu`` sources double as the AMD
    compiled path: PyTorch's ``load_inline`` auto-hipifies the source. NVCC's
    fast-math and explicit FTZ options are translated to HIP/Clang spellings
    before cache fingerprinting; callers need no separate HIP loader.

    Environment knobs:
      - ``HYDROFORGE_CUDA_REBUILD=1``: force rebuild.
      - ``HYDROFORGE_CUDA_USE_SYSTEM_COMPILER=always|never``: explicit host compiler.
    Every binary is stored in the one canonical fingerprint directory. There
    is no host-scoped or flat-layout compatibility lookup.

    Calls in one process are serialized because the compiler environment is
    process-global. Independent processes still coordinate per fingerprint.
    This guards this loader's changes, not arbitrary external environment writes.

    ``cache_only`` returns a matching module or None without starting a build
    or waiting for a compile lock. Forced rebuilds are treated as cache misses.
    """
    from torch.utils.cpp_extension import _get_build_directory, load_inline

    env_kind = "CUDA"
    if force_rebuild is None:
        force_rebuild = _env_truthy(
            f"{env_prefix}_{env_kind}_REBUILD", False
        ) or _env_truthy(f"HYDROFORGE_{env_kind}_REBUILD", False)
    if system_compiler is None:
        system_compiler = os.environ.get(
            f"{env_prefix}_{env_kind}_USE_SYSTEM_COMPILER",
            os.environ.get(f"HYDROFORGE_{env_kind}_USE_SYSTEM_COMPILER", "always"),
        )
    compiler_selection = _resolve_system_compiler(system_compiler)

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

    if cache_only:
        mod = None if force_rebuild else _try_import_cached(
            compiled_name, digest, (cache_root / digest[:16],),
        )
        if mod is not None:
            _module_cache[cache_key] = mod
        return mod

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
        with _temporary_build_environment(compiler_selection):
            if shutil.which("ninja") is None:
                raise RuntimeError(
                    "Ninja is unavailable to the active Python interpreter "
                    f"{sys.executable!r}; install it with "
                    f"{sys.executable!r} -m pip install ninja"
                )
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
        atomic_write_text(
            build_dir / f"{compiled_name}.srchash",
            digest,
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

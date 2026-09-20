"""Parallel precompilation orchestration for declarative CUDA extensions."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from collections import deque
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from hydroforge.contracts.errors import ResourceCleanupError
from hydroforge.data.distributed import get_local_process_rank
from hydroforge.kernels.backends.cuda.build import (
    _safe_path_component,
    load_inline_cu_module,
)
from hydroforge.kernels.backends.cuda.spec import _CompiledCudaExtension

_WORKER_STOP_TIMEOUT = 5.0


@dataclass(slots=True)
class _PrecompileWorker:
    label: str
    process: subprocess.Popen
    log_path: Path
    payload_path: Path
    process_group: int | None = None


def _signal_precompile_session(leader: int, signum: int) -> tuple[BaseException, ...]:
    """Include Ninja's separate command groups in the owned Linux session."""

    groups = {leader}
    failures: list[BaseException] = []
    if sys.platform == "linux":
        try:
            for path in Path("/proc").glob("[0-9]*/stat"):
                try:
                    fields = path.read_bytes().rsplit(b") ", 1)[1].split()
                except (FileNotFoundError, ProcessLookupError, PermissionError):
                    continue
                if int(fields[3]) == leader:
                    groups.add(int(fields[2]))
        except BaseException as error:
            failures.append(error)
    for group in groups:
        try:
            os.killpg(group, signum)
        except ProcessLookupError:
            pass
        except BaseException as error:
            failures.append(error)
    return tuple(failures)


def _stop_precompile_workers(
    workers: Iterable[_PrecompileWorker],
) -> tuple[BaseException, ...]:
    """Stop owned compiler groups, reap direct workers and unlink their files."""

    owned = tuple(workers)
    failures: list[BaseException] = []
    for worker in owned:
        try:
            if worker.process_group is not None:
                failures.extend(
                    _signal_precompile_session(worker.process_group, signal.SIGTERM)
                )
            elif worker.process.poll() is None:
                worker.process.terminate()
        except BaseException as error:
            failures.append(error)
    for worker in owned:
        try:
            try:
                worker.process.wait(timeout=_WORKER_STOP_TIMEOUT)
            except subprocess.TimeoutExpired:
                worker.process.kill()
                worker.process.wait(timeout=_WORKER_STOP_TIMEOUT)
        except BaseException as error:
            failures.append(error)
        if worker.process_group is not None:
            failures.extend(
                _signal_precompile_session(worker.process_group, signal.SIGKILL)
            )
        for path in (worker.log_path, worker.payload_path):
            try:
                path.unlink(missing_ok=True)
            except BaseException as error:
                failures.append(error)
    return tuple(failures)


def _cleanup_precompile_files(
    scope: str,
    primary: BaseException,
    paths: Iterable[Path],
    *,
    stream: Any = None,
) -> None:
    """Attempt every file cleanup, preserving the failure that initiated it."""
    failures = [primary]
    if stream is not None:
        try:
            stream.close()
        except BaseException as error:
            failures.append(error)
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except BaseException as error:
            failures.append(error)
    if len(failures) > 1:
        raise ResourceCleanupError(scope, failures) from primary


def _compile_extension_payload(payload_path: str) -> None:
    """Compile one fully serialized declaration in an isolated worker."""
    path = Path(payload_path)
    try:
        serialized = path.read_text()
    except BaseException as primary:
        _cleanup_precompile_files("CUDA precompile payload read", primary, (path,))
        raise
    path.unlink(missing_ok=True)
    load_inline_cu_module(**json.loads(serialized))


def _precompile_jobs(count: int, default_jobs: int) -> int:
    configured_jobs = os.environ.get("HYDROFORGE_PRECOMPILE_JOBS")
    if configured_jobs is None:
        jobs = 1 if get_local_process_rank() != 0 else default_jobs
    else:
        try:
            jobs = int(configured_jobs)
        except ValueError as error:
            raise ValueError(
                "HYDROFORGE_PRECOMPILE_JOBS must be a positive integer, "
                f"got {configured_jobs!r}"
            ) from error
    if type(jobs) is not int or jobs < 1:
        name = (
            "default_jobs" if configured_jobs is None else "HYDROFORGE_PRECOMPILE_JOBS"
        )
        raise ValueError(f"{name} must be a positive integer, got {jobs!r}")
    return min(jobs, count)


def precompile_extension_specs(
    binary_prefix: str,
    specs: Mapping[str, _CompiledCudaExtension],
    *,
    env_prefix: str = "HYDROFORGE",
    default_jobs: int = 6,
) -> None:
    """Precompile immutable construction-time CUDA extension plans."""
    if not specs:
        return
    jobs = _precompile_jobs(len(specs), default_jobs)
    requests = tuple(
        (name, spec.loader_arguments(f"{binary_prefix}_{name}", env_prefix))
        for name, spec in specs.items()
    )
    _precompile_uncached_requests(requests, jobs=jobs)


def precompile_cuda_requests(
    requests: Iterable[dict[str, Any]],
    *,
    default_jobs: int = 6,
) -> None:
    """Build the exact validated specializations of an operator program."""
    unique = {
        json.dumps(request, sort_keys=True, separators=(",", ":")): request
        for request in requests
    }
    if not unique:
        return
    jobs = _precompile_jobs(len(unique), default_jobs)
    _precompile_uncached_requests(
        tuple((request["name"], request) for request in unique.values()),
        jobs=jobs,
    )


def _precompile_uncached_requests(
    requests: Sequence[tuple[str, dict[str, Any]]],
    *,
    jobs: int,
) -> None:
    pending = tuple(
        (label, request)
        for label, request in requests
        if load_inline_cu_module(**request, cache_only=True) is None
    )
    if not pending:
        return
    _run_precompile_requests(
        pending,
        jobs=min(jobs, len(pending)),
    )


def _run_precompile_requests(
    requests: Sequence[tuple[str, dict[str, Any]]],
    *,
    jobs: int,
) -> None:
    if jobs == 1:
        for _label, arguments in requests:
            load_inline_cu_module(**arguments)
        return

    pending = deque(requests)
    running: list[_PrecompileWorker] = []
    pythonpath = os.pathsep.join([path for path in sys.path if path])

    def start(label: str, payload: dict[str, Any]) -> None:
        env = os.environ.copy()
        env["HYDROFORGE_PRECOMPILE_JOBS"] = "1"
        env["PYTHONPATH"] = pythonpath
        prefix = f"hydroforge_cuda_{_safe_path_component(label)}_"
        payload_file = tempfile.NamedTemporaryFile(
            mode="w",
            prefix=prefix,
            suffix=".json",
            delete=False,
        )
        payload_path = Path(payload_file.name)
        try:
            json.dump(payload, payload_file)
            payload_file.close()
        except BaseException as primary:
            _cleanup_precompile_files(
                "CUDA precompile payload creation",
                primary,
                (payload_path,),
                stream=payload_file,
            )
            raise
        code = (
            "from hydroforge.kernels.backends.cuda.precompile import "
            "_compile_extension_payload;"
            f"_compile_extension_payload({str(payload_path)!r})"
        )
        try:
            log = tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=prefix,
                suffix=".log",
                delete=False,
            )
        except BaseException as primary:
            _cleanup_precompile_files(
                "CUDA precompile log creation", primary, (payload_path,)
            )
            raise
        path = Path(log.name)
        try:
            process = subprocess.Popen(
                [sys.executable, "-c", code],
                stdout=log,
                stderr=subprocess.STDOUT,
                env=env,
                start_new_session=os.name == "posix",
            )
        except BaseException as primary:
            _cleanup_precompile_files(
                "CUDA precompile worker start",
                primary,
                (path, payload_path),
                stream=log,
            )
            raise
        worker = _PrecompileWorker(
            label,
            process,
            path,
            payload_path,
            process.pid if os.name == "posix" else None,
        )
        running.append(worker)
        try:
            log.close()
        except BaseException as primary:
            running.remove(worker)
            failures = _stop_precompile_workers((worker,))
            if failures:
                error = ResourceCleanupError(
                    "CUDA precompile worker start",
                    (primary, *failures),
                )
                raise error from primary
            raise

    try:
        while pending or running:
            while pending and len(running) < jobs:
                start(*pending.popleft())
            for item in tuple(running):
                if item.process.poll() is None:
                    continue
                if item.process.returncode:
                    output = item.log_path.read_text(errors="replace")
                    raise RuntimeError(
                        f"CUDA extension {item.label!r} failed:\n{output}"
                    )
                item.log_path.unlink(missing_ok=True)
                item.payload_path.unlink(missing_ok=True)
                running.remove(item)
            if running:
                time.sleep(0.1)
    except BaseException as primary:
        cleanup_failures = _stop_precompile_workers(running)
        if cleanup_failures:
            error = ResourceCleanupError(
                "CUDA precompile workers",
                (primary, *cleanup_failures),
            )
            raise error from primary
        raise


def precompile_cuda_modules(
    module_names: Iterable[str],
    *,
    opened_modules: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Precompile every CUDA catalog nominally owned by each module.

    A downstream CUDA adapter already declares all extensions through
    :class:`CudaExtensionGroup`; requiring a second, specially named forwarding
    hook repeats that information and makes discovery depend on spelling.
    """

    from hydroforge.kernels.backends.cuda.dispatcher import CudaExtensionGroup

    if opened_modules is not None:
        opened_modules = tuple(opened_modules)
    results: dict[str, Any] = {}
    for module_name in module_names:
        mod = importlib.import_module(module_name)
        groups = []
        seen_groups: set[int] = set()
        for value in vars(mod).values():
            if (
                isinstance(value, CudaExtensionGroup)
                and value.owner_module == module_name
                and id(value) not in seen_groups
            ):
                groups.append(value)
                seen_groups.add(id(value))
        if not groups:
            foreign = sorted(
                {
                    value.owner_module
                    for value in vars(mod).values()
                    if isinstance(value, CudaExtensionGroup)
                }
            )
            detail = f"; imported owners={foreign}" if foreign else ""
            raise ValueError(
                f"{module_name} declares no owned CudaExtensionGroup{detail}"
            )
        prefixes = [group.binary_prefix for group in groups]
        if len(prefixes) != len(set(prefixes)):
            raise ValueError(
                f"{module_name} has duplicate CUDA binary prefixes: {prefixes}"
            )
        results[module_name] = {
            group.binary_prefix: (
                group._ensure_precompiled()
                if opened_modules is None
                else group._ensure_precompiled_for_modules(opened_modules)
            )
            for group in groups
        }
    return results


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="hydroforge-cuda-precompile",
        description="Precompile CUDA extensions declared by hydroforge model modules.",
    )
    parser.add_argument(
        "modules",
        nargs="+",
        help=(
            "Python modules declaring owned CudaExtensionGroup catalogs, "
            "e.g. cmfgpu.phys.cuda"
        ),
    )
    args = parser.parse_args(argv)
    precompile_cuda_modules(args.modules)
    return 0

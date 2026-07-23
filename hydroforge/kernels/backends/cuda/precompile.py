"""Parallel precompilation orchestration for declarative CUDA extensions."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import importlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

from hydroforge.contracts import ResourceCleanupError
from hydroforge.data.distributed import get_local_process_rank
from hydroforge.kernels.backends.cuda.build import _safe_path_component, load_inline_cu_module
from hydroforge.kernels.backends.cuda.spec import CudaExtensionSpec, cuda_declarations


_WORKER_STOP_TIMEOUT = 5.0


@dataclass(slots=True)
class _PrecompileWorker:
    label: str
    process: subprocess.Popen
    log_path: Path
    payload_path: Path


def _stop_precompile_workers(
    workers: Iterable[_PrecompileWorker],
) -> tuple[BaseException, ...]:
    """Terminate, reap and unlink every outstanding compiler worker."""

    owned = tuple(workers)
    failures: list[BaseException] = []
    for worker in owned:
        try:
            if worker.process.poll() is None:
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
        for path in (worker.log_path, worker.payload_path):
            try:
                path.unlink(missing_ok=True)
            except BaseException as error:
                failures.append(error)
    return tuple(failures)


def _compile_extension_payload(payload_path: str) -> None:
    """Compile one fully serialized declaration in an isolated worker."""
    path = Path(payload_path)
    try:
        serialized = path.read_text()
    finally:
        path.unlink(missing_ok=True)
    payload = json.loads(serialized)
    source = payload["materialized_source"]
    declarations = payload["declarations"] or cuda_declarations(
        source, payload["functions"],
    )
    load_inline_cu_module(
        payload["name"],
        cpp_sources="\n".join((*payload["cpp_headers"], *declarations)),
        cuda_sources=source,
        functions=payload["functions"],
        extra_cuda_cflags=payload["cflags"],
        extra_include_paths=payload["include_paths"],
        extra_ldflags=payload["ldflags"],
        env_prefix=payload["env_prefix"],
    )


def precompile_extension_specs(
    binary_prefix: str,
    specs: Mapping[str, CudaExtensionSpec],
    *,
    env_prefix: str = "HYDROFORGE",
    default_jobs: int = 6,
    materialized_sources: Mapping[str, str] | None = None,
) -> None:
    """Precompile declarative specs without mutating their owner module."""
    if type(default_jobs) is not int or default_jobs < 1:
        raise ValueError("default_jobs must be an exact positive int")
    if type(binary_prefix) is not str or not binary_prefix.isidentifier():
        raise ValueError("CUDA precompile binary_prefix must be an identifier")
    if type(env_prefix) is not str or not env_prefix.isidentifier():
        raise ValueError("CUDA precompile env_prefix must be an identifier")
    if not isinstance(specs, Mapping):
        raise TypeError("CUDA precompile specs must be a mapping")
    invalid_specs = {
        name: type(spec).__name__
        for name, spec in specs.items()
        if type(name) is not str or not name.isidentifier()
        or not isinstance(spec, CudaExtensionSpec)
    }
    if invalid_specs:
        raise TypeError(
            f"invalid CUDA precompile extension specs: {invalid_specs}"
        )
    if not specs:
        return
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
        if jobs < 1:
            raise ValueError(
                "HYDROFORGE_PRECOMPILE_JOBS must be a positive integer, "
                f"got {configured_jobs!r}"
            )
    jobs = min(jobs, len(specs))
    if materialized_sources is None:
        sources = {
            name: spec.materialize_source() for name, spec in specs.items()
        }
    else:
        if set(materialized_sources) != set(specs):
            raise ValueError(
                "materialized CUDA sources must exactly match extension specs: "
                f"missing={sorted(set(specs).difference(materialized_sources))}, "
                f"extra={sorted(set(materialized_sources).difference(specs))}"
            )
        invalid = {
            name: type(source).__name__
            for name, source in materialized_sources.items()
            if type(source) is not str or not source
        }
        if invalid:
            raise TypeError(
                "materialized CUDA sources must be non-empty strings: "
                f"{invalid}"
            )
        sources = dict(materialized_sources)
    if jobs == 1:
        for name, spec in specs.items():
            source = sources[name]
            load_inline_cu_module(
                f"{binary_prefix}_{name}",
                cpp_sources="\n".join((
                    *spec.cpp_headers,
                    *(spec.declarations or cuda_declarations(source, spec.functions)),
                )),
                cuda_sources=source,
                functions=spec.functions,
                extra_cuda_cflags=spec.cflags,
                extra_include_paths=tuple(map(str, spec.include_paths)),
                extra_ldflags=spec.ldflags,
                env_prefix=env_prefix,
            )
        return

    pending = []
    for name, spec in specs.items():
        payload = {
            "name": f"{binary_prefix}_{name}",
            "materialized_source": sources[name],
            "functions": spec.functions,
            "declarations": spec.declarations,
            "cflags": spec.cflags,
            "cpp_headers": spec.cpp_headers,
            "include_paths": tuple(map(str, spec.include_paths)),
            "ldflags": spec.ldflags,
            "env_prefix": env_prefix,
        }
        pending.append((name, payload))
    running: list[_PrecompileWorker] = []
    pythonpath = os.pathsep.join([path for path in sys.path if path])

    def start(label: str, payload: dict[str, Any]) -> None:
        env = os.environ.copy()
        env["HYDROFORGE_PRECOMPILE_JOBS"] = "1"
        env["PYTHONPATH"] = pythonpath
        payload_file = tempfile.NamedTemporaryFile(
            mode="w",
            prefix=f"hydroforge_cuda_{_safe_path_component(label)}_",
            suffix=".json",
            delete=False,
        )
        payload_path = Path(payload_file.name)
        try:
            json.dump(payload, payload_file)
            payload_file.close()
        except BaseException as primary:
            failures: list[BaseException] = []
            try:
                payload_file.close()
            except BaseException as error:
                failures.append(error)
            try:
                payload_path.unlink(missing_ok=True)
            except BaseException as error:
                failures.append(error)
            if failures:
                error = ResourceCleanupError(
                    "CUDA precompile payload creation", (primary, *failures),
                )
                raise error from primary
            raise
        code = (
            "from hydroforge.kernels.backends.cuda.precompile import "
            "_compile_extension_payload;"
            f"_compile_extension_payload({str(payload_path)!r})"
        )
        try:
            log = tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f"hydroforge_cuda_{_safe_path_component(label)}_",
                suffix=".log",
                delete=False,
            )
        except BaseException:
            payload_path.unlink(missing_ok=True)
            raise
        path = Path(log.name)
        try:
            process = subprocess.Popen(
                [sys.executable, "-c", code], stdout=log,
                stderr=subprocess.STDOUT, env=env,
            )
        except BaseException as primary:
            failures: list[BaseException] = []
            try:
                log.close()
            except BaseException as error:
                failures.append(error)
            try:
                path.unlink(missing_ok=True)
            except BaseException as error:
                failures.append(error)
            try:
                payload_path.unlink(missing_ok=True)
            except BaseException as error:
                failures.append(error)
            if failures:
                error = ResourceCleanupError(
                    "CUDA precompile worker start", (primary, *failures),
                )
                raise error from primary
            raise
        worker = _PrecompileWorker(
            label, process, path, payload_path,
        )
        running.append(worker)
        try:
            log.close()
        except BaseException as primary:
            running.remove(worker)
            failures = _stop_precompile_workers((worker,))
            if failures:
                error = ResourceCleanupError(
                    "CUDA precompile worker start", (primary, *failures),
                )
                raise error from primary
            raise

    try:
        while pending or running:
            while pending and len(running) < jobs:
                start(*pending.pop(0))
            for item in tuple(running):
                if item.process.poll() is None:
                    continue
                output = item.log_path.read_text(errors="replace")
                item.log_path.unlink(missing_ok=True)
                item.payload_path.unlink(missing_ok=True)
                running.remove(item)
                if item.process.returncode:
                    raise RuntimeError(
                        f"CUDA extension {item.label!r} failed:\n{output}"
                    )
            if running:
                time.sleep(0.1)
    except BaseException as primary:
        cleanup_failures = _stop_precompile_workers(running)
        if cleanup_failures:
            error = ResourceCleanupError(
                "CUDA precompile workers", (primary, *cleanup_failures),
            )
            raise error from primary
        raise


def precompile_cuda_modules(module_names: Iterable[str]) -> Dict[str, Any]:
    """Precompile every CUDA catalog nominally owned by each module.

    A downstream CUDA adapter already declares all extensions through
    :class:`CudaExtensionGroup`; requiring a second, specially named forwarding
    hook repeats that information and makes discovery depend on spelling.
    """

    from hydroforge.kernels.backends.cuda.dispatcher import CudaExtensionGroup

    results: Dict[str, Any] = {}
    for module_name in module_names:
        mod = importlib.import_module(module_name)
        groups = tuple(dict.fromkeys(
            value
            for value in vars(mod).values()
            if isinstance(value, CudaExtensionGroup)
            and value.owner_module == module_name
        ))
        if not groups:
            foreign = sorted({
                value.owner_module
                for value in vars(mod).values()
                if isinstance(value, CudaExtensionGroup)
            })
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
            group.binary_prefix: group.ensure_precompiled()
            for group in groups
        }
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
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

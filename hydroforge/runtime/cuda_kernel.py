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

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import json as _json

import torch

_module_cache: Dict[str, Any] = {}

_MANIFEST_FILENAME = "compile_manifest.json"


def load_build_manifest(build_dir: Union[str, Path]) -> dict:
    """Load the compile manifest from a build directory."""
    p = Path(build_dir) / _MANIFEST_FILENAME
    if p.exists():
        with open(p) as f:
            return _json.load(f)
    return {}


def update_build_manifest(
    build_dir: Union[str, Path], section: str, data: dict
) -> None:
    """Update one section of the compile manifest and write it back."""
    manifest = load_build_manifest(build_dir)
    manifest[section] = data
    p = Path(build_dir) / _MANIFEST_FILENAME
    with open(p, "w") as f:
        _json.dump(manifest, f, indent=2)


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

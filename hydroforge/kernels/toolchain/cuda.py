# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Runtime compilation and launch of CUDA/HIP device code.

NVRTC (hiprtc under ROCm) compiles device-only sources in process, so no host
compiler, CPython binding or ATen headers are parsed. A program compiles only
the kernel instantiations its launch plans name; the binary is cached in memory
and on disk by content. Launches pass pre-packed arguments to the driver on the
caller's current stream, which keeps them valid under CUDA Graph capture.

The runtime compiler and the driver are bound through ``ctypes`` only, which
serves HIP as well; conditional WHILE graphs use the same driver binding.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import logging
import math
import re
import sys
import threading
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from functools import cache, lru_cache, partial
from pathlib import Path
from typing import Any

import torch

from hydroforge.core.errors import cleanup_on_exit
from hydroforge.io.files import atomic_output_path, atomic_write_text
from hydroforge.kernels.codegen.types import ELEMENTS, MANGLED_ELEMENTS, element
from hydroforge.kernels.toolchain import CompileRequest, amdgpu, mangling
from hydroforge.kernels.toolchain.cache import (
    acquire_compile_lock,
    release_compile_lock,
)
from hydroforge.platform import env
from hydroforge.platform.backend import CUDA

# Device code sees fixed-width integers, the <cmath> constants and a portable
# full-warp mask without any host header; everything else comes from libcu++
# (cuda::std) or builtins.
RTC_PRELUDE = r"""#ifndef HYDROFORGE_RTC_PRELUDE
#define HYDROFORGE_RTC_PRELUDE
#if defined(__HIPCC_RTC__)
using __hip_internal::int8_t; using __hip_internal::int16_t;
using __hip_internal::int32_t; using __hip_internal::int64_t;
using __hip_internal::uint8_t; using __hip_internal::uint16_t;
using __hip_internal::uint32_t; using __hip_internal::uint64_t;
#ifndef INFINITY
#define INFINITY (__builtin_huge_valf())
#endif
#ifndef NAN
#define NAN (__builtin_nanf(""))
#endif
#else
#include <cuda/std/cstdint>
#include <cuda/std/limits>
using cuda::std::int8_t; using cuda::std::int16_t;
using cuda::std::int32_t; using cuda::std::int64_t;
using cuda::std::uint8_t; using cuda::std::uint16_t;
using cuda::std::uint32_t; using cuda::std::uint64_t;
#ifndef INFINITY
#define INFINITY (::cuda::std::numeric_limits<float>::infinity())
#endif
#ifndef NAN
#define NAN (::cuda::std::numeric_limits<float>::quiet_NaN())
#endif
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// Full-warp mask for the *_sync intrinsics: HIP requires 64 bits and narrows
// it on wave32 devices.
#if defined(__HIPCC_RTC__)
#define HYDROFORGE_WARP_MASK 0xffffffffffffffffull
#else
#define HYDROFORGE_WARP_MASK 0xffffffffu
#endif
#endif
"""

_logger = logging.getLogger(__name__)
_CACHE_VERSION = 2
_INCLUDES = re.compile(
    r"(?:#|%:)\s*(?:/\*.*?\*/\s*)*(?:include(?:_next)?|import|embed)\b([^\n]*)",
    re.DOTALL,
)


# ---------------------------------------------------------------------- #
# Toolkit
# ---------------------------------------------------------------------- #


class _Toolkit:
    """ctypes view of the runtime compiler and driver matching PyTorch."""

    def __init__(self) -> None:
        self.hip = torch.version.hip is not None
        if self.hip:
            self.rtc = _load_first(("libhiprtc.so", "libhiprtc.so.7", "libhiprtc.so.6"))
            self.driver = _load_first(
                ("libamdhip64.so", "libamdhip64.so.7", "libamdhip64.so.6")
            )
            self._rtc_prefix, self._driver_prefix = "hiprtc", "hip"
        else:
            major = torch.version.cuda.split(".")[0]
            names = (
                (f"nvrtc64_{major}0_0.dll",)
                if sys.platform == "win32"
                else (f"libnvrtc.so.{major}", "libnvrtc.so")
            )
            self.rtc = _load_first(names)
            self.driver = ctypes.CDLL(
                "nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1"
            )
            self._rtc_prefix, self._driver_prefix = "nvrtc", "cu"
        p, s, i, sz = ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_size_t
        self._bind_rtc("CreateProgram", ctypes.POINTER(p), s, s, i, p, p)
        self._bind_rtc("AddNameExpression", p, s)
        self._bind_rtc("CompileProgram", p, i, ctypes.POINTER(s))
        self._bind_rtc("GetProgramLogSize", p, ctypes.POINTER(sz))
        self._bind_rtc("GetProgramLog", p, s)
        self._bind_rtc("GetLoweredName", p, s, ctypes.POINTER(s))
        self._bind_rtc("DestroyProgram", ctypes.POINTER(p))
        self._bind_rtc("Version", ctypes.POINTER(i), ctypes.POINTER(i))
        self.rtc_error = getattr(self.rtc, f"{self._rtc_prefix}GetErrorString")
        self.rtc_error.restype = s
        binary = "Code" if self.hip else "CUBIN"
        self._bind_rtc(f"Get{binary}Size", p, ctypes.POINTER(sz), alias="GetBinarySize")
        self._bind_rtc(f"Get{binary}", p, s, alias="GetBinary")
        if not self.hip:
            self._bind_rtc("GetPTXSize", p, ctypes.POINTER(sz))
            self._bind_rtc("GetPTX", p, s)
        self._bind_driver("ModuleLoadData", ctypes.POINTER(p), p)
        self._bind_driver("ModuleGetFunction", ctypes.POINTER(p), p, s)
        launch = "ModuleLaunchKernel" if self.hip else "LaunchKernel"
        self._bind_driver(
            launch,
            p,
            *(ctypes.c_uint,) * 6,
            ctypes.c_uint,
            p,
            p,
            p,
            alias="LaunchKernel",
        )
        self._bind_driver("FuncSetAttribute", p, i, i)
        self.param_info = None
        if not self.hip and hasattr(self.driver, "cuFuncGetParamInfo"):
            self._bind_driver(
                "FuncGetParamInfo",
                p,
                sz,
                ctypes.POINTER(sz),
                ctypes.POINTER(sz),
                alias="param_info",
            )
        error = getattr(self.driver, f"{self._driver_prefix}GetErrorString")
        if self.hip:
            error.argtypes, error.restype = (i,), s
        else:
            error.argtypes, error.restype = (i, ctypes.POINTER(s)), i
        self._driver_error = error
        major, minor = ctypes.c_int(), ctypes.c_int()
        self.check_rtc(self.Version(ctypes.byref(major), ctypes.byref(minor)))
        self.version = (major.value, minor.value)
        self.driver_version = 0
        if not self.hip:
            self._bind_driver("DriverGetVersion", ctypes.POINTER(i))
            version = ctypes.c_int()
            self.check_driver(
                self.DriverGetVersion(ctypes.byref(version)), "querying the driver"
            )
            self.driver_version = version.value

    def _bind_rtc(self, symbol: str, *argtypes: Any, alias: str | None = None) -> None:
        function = _bind(self.rtc, f"{self._rtc_prefix}{symbol}", *argtypes)
        setattr(self, alias or symbol, function)

    def _bind_driver(
        self, symbol: str, *argtypes: Any, alias: str | None = None
    ) -> None:
        function = _bind(self.driver, f"{self._driver_prefix}{symbol}", *argtypes)
        setattr(self, alias or symbol, function)

    def check_rtc(self, result: int) -> None:
        if result != 0:
            raise RuntimeError(
                f"runtime compiler error: {self.rtc_error(result).decode()}"
            )

    def check_driver(self, result: int, action: str) -> None:
        if result == 0:
            return
        if self.hip:
            message = self._driver_error(result)
        else:
            text = ctypes.c_char_p()
            self._driver_error(result, ctypes.byref(text))
            message = text.value
        raise RuntimeError(
            f"{action} failed: {message.decode() if message else f'error {result}'}"
        )


def _bind(library: ctypes.CDLL, symbol: str, *argtypes: Any) -> Any:
    """One entry point of ``library`` returning a status code."""
    function = getattr(library, symbol)
    function.argtypes, function.restype = argtypes, ctypes.c_int
    return function


def _load_first(names: Sequence[str]) -> ctypes.CDLL:
    errors = []
    for name in names:
        try:
            return ctypes.CDLL(name)
        except OSError as error:
            errors.append(f"{name}: {error}")
    raise OSError("runtime compiler library unavailable: " + "; ".join(errors))


@cache
def toolkit() -> _Toolkit:
    return _Toolkit()


# ---------------------------------------------------------------------- #
# Programs and compilation
# ---------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class RtcProgram:
    """Immutable device-only source and its runtime-compiler options."""

    source: str
    options: tuple[str, ...] = ()
    name: str = "hydroforge"


def program_options(options: tuple[str, ...], *, physics: bool) -> tuple[str, ...]:
    """Runtime-compiler options of a program under the math policy.

    Physics programs follow the math mode.  Framework programs (statistics,
    step fields, loop control) never contract multiplies and adds, so their
    results do not depend on how the generated code is structured.
    """

    if not physics:
        return ("--fmad=false", *options)
    if CUDA.math.physics_fast_math and "--use_fast_math" not in options:
        return ("--use_fast_math", *options)
    return options


@dataclass(frozen=True, slots=True)
class RtcRequest:
    """One program with the exact kernel instantiations a launch plan names."""

    program: RtcProgram
    kernels: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.kernels:
            raise ValueError(f"{self.program.name}: a runtime program needs kernels")
        object.__setattr__(self, "kernels", tuple(sorted(set(self.kernels))))


@dataclass(frozen=True, slots=True)
class _Binary:
    """A compiled image, its lowered kernel names and its cache key."""

    image: bytes
    lowered: dict[str, str]
    key: str


def _target(device: int) -> tuple[str, bool]:
    """Return the compile target and whether it is a device binary.

    A GPU newer than the runtime compiler receives PTX of the newest supported
    architecture, which the driver lowers on load.
    """
    properties = torch.cuda.get_device_properties(device)
    if toolkit().hip:
        return properties.gcnArchName, True
    arch = properties.major * 10 + properties.minor
    supported = _supported_architectures()
    if arch in supported:
        return f"sm_{arch}", True
    return f"compute_{max(a for a in supported if a <= arch)}", False


@cache
def _supported_architectures() -> frozenset[int]:
    kit = toolkit()
    count = ctypes.c_int()
    number = _bind(kit.rtc, "nvrtcGetNumSupportedArchs", ctypes.POINTER(ctypes.c_int))
    kit.check_rtc(number(ctypes.byref(count)))
    values = (ctypes.c_int * count.value)()
    listing = _bind(kit.rtc, "nvrtcGetSupportedArchs", ctypes.POINTER(ctypes.c_int))
    kit.check_rtc(listing(values))
    return frozenset(values)


@lru_cache(maxsize=1024)
def _key(request: RtcRequest, target: str) -> str:
    """Cache identity, memoized: hashing a large source is not repeated."""
    kit = toolkit()
    identity = {
        "version": _CACHE_VERSION,
        "source": RTC_PRELUDE + request.program.source,
        "name": request.program.name,
        "options": _options(request.program, target),
        "kernels": list(request.kernels),
        "target": target,
        "compiler": [("hip" if kit.hip else "cuda"), *kit.version],
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _cache_root() -> Path:
    from torch.utils.cpp_extension import _get_build_directory

    return Path(_get_build_directory("hydroforge_rtc", verbose=False))


def _compile(
    request: RtcRequest, target: str, device_binary: bool, key: str
) -> _Binary:
    kit = toolkit()
    program = ctypes.c_void_p()
    source = RTC_PRELUDE + request.program.source
    kit.check_rtc(
        kit.CreateProgram(
            ctypes.byref(program),
            source.encode(),
            f"{request.program.name}.cu".encode(),
            0,
            None,
            None,
        )
    )
    with cleanup_on_exit(
        f"{request.program.name} runtime compiler program",
        (lambda: kit.check_rtc(kit.DestroyProgram(ctypes.byref(program))),),
    ):
        for kernel in request.kernels:
            kit.check_rtc(kit.AddNameExpression(program, kernel.encode()))
        options = _options(request.program, target)
        encoded = (ctypes.c_char_p * len(options))(*(o.encode() for o in options))
        result = kit.CompileProgram(program, len(options), encoded)
        if result != 0:
            raise RuntimeError(
                f"{request.program.name}: runtime compilation failed\n"
                f"{_program_log(program)}"
            )
        size = ctypes.c_size_t()
        if device_binary:
            kit.check_rtc(kit.GetBinarySize(program, ctypes.byref(size)))
            image = ctypes.create_string_buffer(size.value)
            kit.check_rtc(kit.GetBinary(program, image))
        else:
            kit.check_rtc(kit.GetPTXSize(program, ctypes.byref(size)))
            image = ctypes.create_string_buffer(size.value)
            kit.check_rtc(kit.GetPTX(program, image))
        lowered = {}
        for kernel in request.kernels:
            name = ctypes.c_char_p()
            kit.check_rtc(
                kit.GetLoweredName(program, kernel.encode(), ctypes.byref(name))
            )
            lowered[kernel] = name.value.decode()
        return _Binary(image.raw, lowered, key)


def _program_log(program: ctypes.c_void_p) -> str:
    kit = toolkit()
    size = ctypes.c_size_t()
    try:
        kit.check_rtc(kit.GetProgramLogSize(program, ctypes.byref(size)))
        log = ctypes.create_string_buffer(size.value)
        kit.check_rtc(kit.GetProgramLog(program, log))
    except RuntimeError as error:
        return f"<compilation log unavailable: {error}>"
    return log.value.decode(errors="replace")


# hiprtc spells the numerical controls as Clang flags; ptxas has no AMD analog.
# NVRTC fast math never assumes finite values, so Clang keeps honoring NaN and
# infinity: under ``-ffinite-math-only`` it folds ``isnan`` checks to false.
_HIP_OPTIONS = {
    "--use_fast_math": ("-ffast-math", "-fno-finite-math-only"),
    "--ftz=false": ("-fno-gpu-flush-denormals-to-zero",),
    "--ftz=true": ("-fgpu-flush-denormals-to-zero",),
    "--fmad=false": ("-ffp-contract=off",),
    "--fmad=true": ("-ffp-contract=fast",),
}


def _options(program: RtcProgram, target: str) -> tuple[str, ...]:
    """The exact options shared by cache identity and runtime compilation."""
    if toolkit().hip:
        arch = f"--offload-arch={target}"
        options = tuple(
            flag
            for option in program.options
            if not option.startswith("--ptxas-options")
            for flag in _HIP_OPTIONS.get(option, (option,))
        )
    else:
        arch, options = f"--gpu-architecture={target}", program.options
    return (arch, "--std=c++20", *_include_options(), *options)


@cache
def _toolkit_include_roots() -> tuple[Path, ...]:
    """The CUDA toolkit's include directories, as PyTorch locates them."""
    from torch.utils.cpp_extension import include_paths

    return tuple(dict.fromkeys(Path(path) for path in include_paths("cuda")))


@cache
def _include_options() -> tuple[str, ...]:
    """Expose libcu++ and the CUDA headers shipped with the toolkit."""
    if toolkit().hip:
        return ()
    roots = _toolkit_include_roots()
    candidates = [*roots, *(root / "cccl" for root in roots)]
    return tuple(
        f"-I{path}" for path in dict.fromkeys(candidates) if (path / "cuda").is_dir()
    )


@cache
def toolkit_include_options() -> tuple[str, ...]:
    """Expose the toolkit's own CUDA headers (e.g. ``cuda_device_runtime_api.h``).

    Only programs that need device-runtime declarations add these; everything
    else compiles against NVRTC's built-in headers and libcu++ alone.
    """
    if toolkit().hip:
        return ()
    return tuple(
        f"-I{path}"
        for path in _toolkit_include_roots()
        if (path / "cuda_device_runtime_api.h").is_file()
    )


_memory: dict[str, _Binary] = {}
_memory_lock = threading.Lock()


def _persistent(program: RtcProgram) -> bool:
    """Only persist programs whose remaining headers belong to the toolkit.

    Quoted application headers are already expanded by ``CudaSource``. NVRTC
    exposes no dependency list for other includes; external search paths,
    forced headers and macro includes therefore keep only a process-local
    snapshot. Scanning entire include trees would add work to every build
    without reliably reproducing the compiler's preprocessing rules.
    """
    toolkit_paths = (*_include_options(), *toolkit_include_options())
    for option in program.options:
        if (
            option.startswith(
                ("-I", "--include", "-include", "--pre-include", "-isystem", "-iquote")
            )
            and option not in toolkit_paths
        ):
            return False
    roots = tuple(Path(option[2:]).resolve() for option in toolkit_paths)
    # RTC_PRELUDE is framework-owned and contains only toolkit/builtin
    # headers, including inactive CUDA includes under HIP's #else branch.
    source = program.source.replace("\\\n", "").replace("\\\r\n", "")
    for include in _INCLUDES.findall(source):
        literal = re.fullmatch(r"\s*<([^>]+)>\s*(?://.*|/\*.*\*/\s*)?", include)
        if literal is None:
            return False
        name = literal.group(1)
        if not any(
            (root / name).is_file() and (root / name).resolve().is_relative_to(root)
            for root in roots
        ):
            return False
    return True


def compile_request(request: RtcRequest, device: int) -> _Binary:
    """Return the cached binary for ``request``, compiling it at most once."""

    target, device_binary = _target(device)
    key = _key(request, target)
    with _memory_lock:
        cached = _memory.get(key)
    if cached is not None:
        return cached
    persistent = _persistent(request.program)
    try:
        directory = _cache_root() / key[:2]
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        return _compile_unshared(request, target, device_binary, key, error)
    image_path = directory / f"{key}.bin"
    names_path = directory / f"{key}.json"

    def load() -> _Binary | None:
        with _memory_lock:
            cached = _memory.get(key)
        if cached is not None:
            return cached
        if (
            not persistent
            or env.flag(env.CUDA_REBUILD, default=False)
            or not (image_path.is_file() and names_path.is_file())
        ):
            return None
        return _Binary(image_path.read_bytes(), json.loads(names_path.read_text()), key)

    binary = load()
    if binary is None:
        lock = directory / f"{key}.lock"
        try:
            token, binary = acquire_compile_lock(lock, cache_probe=load)
        except OSError as error:
            return _compile_unshared(request, target, device_binary, key, error)
        if binary is None:
            with cleanup_on_exit(
                f"{request.program.name} compile lock",
                (lambda: release_compile_lock(lock, token),),
            ):
                # Another thread may have published between the probe and
                # lock acquisition. Publish memory before releasing the lock,
                # including programs that deliberately have no disk artifact.
                binary = load()
                if binary is None:
                    binary = _compile(request, target, device_binary, key)
                    if persistent:
                        try:
                            with atomic_output_path(image_path) as temporary:
                                temporary.write_bytes(binary.image)
                            atomic_write_text(names_path, json.dumps(binary.lowered))
                        except OSError as error:
                            _logger.warning(
                                "%s: runtime compile cache write failed (%s); "
                                "the binary stays in memory",
                                request.program.name,
                                error,
                            )
                with _memory_lock:
                    return _memory.setdefault(key, binary)
    with _memory_lock:
        return _memory.setdefault(key, binary)


def _compile_unshared(
    request: RtcRequest,
    target: str,
    device_binary: bool,
    key: str,
    error: OSError,
) -> _Binary:
    """Compile into the memory cache when the disk cache is unusable."""

    _logger.warning(
        "%s: runtime compile cache unavailable (%s); compiling in memory only",
        request.program.name,
        error,
    )
    binary = _compile(request, target, device_binary, key)
    with _memory_lock:
        return _memory.setdefault(key, binary)


def precompile_request(request: RtcRequest, device: int) -> CompileRequest:
    """Describe the compilation of ``request`` for one device."""

    target, _device_binary = _target(device)
    return CompileRequest(
        CUDA.toolchain,
        _key(request, target),
        len(request.program.source) * len(request.kernels),
        partial(compile_request, request, device),
    )


# ---------------------------------------------------------------------- #
# Launch description
# ---------------------------------------------------------------------- #


@dataclass(frozen=True, slots=True)
class KernelArgument:
    """One by-value kernel parameter with its exact device representation.

    ``element`` records a pointer's tensor dtype so launches can check it
    against the kernel's declared element type.
    """

    ctype: type
    value: Any
    element: torch.dtype | None = None
    owner: Any = field(default=None, compare=False, repr=False)

    def storage(self) -> ctypes._SimpleCData | ctypes.Structure:
        if isinstance(self.value, tuple):
            return self.ctype(*(field.storage() for field in self.value))
        return self.ctype(self.value)


def pointer(tensor: torch.Tensor | None) -> KernelArgument:
    """Device address of a contiguous CUDA tensor, or a null pointer."""

    if tensor is None:
        return KernelArgument(ctypes.c_void_p, None)
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError(
            "runtime kernels take contiguous CUDA tensors, got "
            f"device={tensor.device} contiguous={tensor.is_contiguous()}"
        )
    return KernelArgument(ctypes.c_void_p, tensor.data_ptr(), tensor.dtype, tensor)


def boolean(value: bool) -> KernelArgument:
    if type(value) is not bool:
        raise TypeError("boolean kernel argument must be an exact bool")
    return KernelArgument(ctypes.c_bool, value)


def int32(value: int) -> KernelArgument:
    return KernelArgument(ctypes.c_int32, _checked_int(value, 32))


def uint32(value: int) -> KernelArgument:
    return KernelArgument(ctypes.c_uint32, _checked_uint(value, 32))


def uint64(value: int) -> KernelArgument:
    return KernelArgument(ctypes.c_uint64, _checked_uint(value, 64))


def int64(value: int) -> KernelArgument:
    return KernelArgument(ctypes.c_int64, _checked_int(value, 64))


def float32(value: float) -> KernelArgument:
    """Round a finite host float to FP32, rejecting range loss."""
    if type(value) is not float or not math.isfinite(value):
        raise TypeError("float32 kernel argument must be an exact finite float")
    converted = ctypes.c_float(value).value
    if not math.isfinite(converted) or (value != 0 and converted == 0):
        raise ValueError("float32 kernel argument overflows or underflows")
    return KernelArgument(ctypes.c_float, value)


def float64(value: float) -> KernelArgument:
    if type(value) is not float or not math.isfinite(value):
        raise TypeError("float64 kernel argument must be an exact finite float")
    return KernelArgument(ctypes.c_double, value)


def scalar(value: Any, dtype: torch.dtype) -> KernelArgument:
    """Scalar in the device type of ``dtype`` (e.g. a precision-typed ``Real``)."""

    if dtype not in _SCALAR_ARGUMENTS:
        raise TypeError(f"unsupported scalar dtype {dtype}")
    return _SCALAR_ARGUMENTS[dtype](value)


def struct(*fields: KernelArgument) -> KernelArgument:
    """A by-value aggregate laid out with natural C alignment."""

    layout = type(
        "KernelStruct",
        (ctypes.Structure,),
        {"_fields_": [(f"f{i}", f.ctype) for i, f in enumerate(fields)]},
    )
    return KernelArgument(layout, tuple(fields))


def _checked_int(value: int, bits: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"integer kernel argument required, got {value!r}")
    if not -(2 ** (bits - 1)) <= value < 2 ** (bits - 1):
        raise ValueError(f"int{bits} kernel argument out of range: {value}")
    return value


def _checked_uint(value: int, bits: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"integer kernel argument required, got {value!r}")
    if not 0 <= value < 2**bits:
        raise ValueError(f"uint{bits} kernel argument out of range: {value}")
    return value


_SCALAR_ARGUMENTS = {
    torch.float32: float32,
    torch.float64: float64,
    torch.int32: int32,
    torch.uint32: uint32,
    torch.int64: int64,
    torch.uint64: uint64,
    torch.bool: boolean,
}


def ctype(value: torch.Tensor | torch.dtype) -> str:
    """C++ spelling of a tensor's element type for template arguments."""

    return element(value.dtype if isinstance(value, torch.Tensor) else value, "cuda")


def _dim3(value: int | Sequence[int], what: str) -> tuple[int, int, int]:
    values = (value,) if isinstance(value, int) else tuple(value)
    if not 1 <= len(values) <= 3 or any(
        isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values
    ):
        raise ValueError(f"CUDA {what} must be 1-3 positive ints, got {value!r}")
    if what == "grid" and (values[0] >= 2**31 or any(v > 65535 for v in values[1:])):
        raise ValueError(f"CUDA grid exceeds device limits: {values}")
    if what == "block" and (
        math.prod(values) > 1024
        or any(v > limit for v, limit in zip(values, (1024, 1024, 64)))
    ):
        raise ValueError(
            f"CUDA block exceeds 1024 threads or the (1024, 1024, 64) "
            f"dimension limits: {values}"
        )
    return (*values, *(1,) * (3 - len(values)))


@dataclass(frozen=True, slots=True)
class CudaLaunch:
    """One kernel launch named by a C++ name expression."""

    kernel: str
    grid: int | tuple[int, ...]
    block: int | tuple[int, ...]
    args: tuple[KernelArgument, ...]
    shared_memory: int = 0

    def __post_init__(self) -> None:
        object.__setattr__(self, "grid", _dim3(self.grid, "grid"))
        object.__setattr__(self, "block", _dim3(self.block, "block"))
        object.__setattr__(self, "args", tuple(self.args))
        if type(self.shared_memory) is not int or not 0 <= self.shared_memory < 2**32:
            raise ValueError("dynamic shared memory must be a nonnegative exact uint32")


LaunchStep = CudaLaunch | Callable[[], Any]


def blocks(count: int, threads: int) -> int:
    """Number of ``threads``-sized blocks covering ``count`` items."""

    return (count - 1) // threads + 1


# ---------------------------------------------------------------------- #
# Loading and prepared launches
# ---------------------------------------------------------------------- #

_modules: dict[tuple[str, int], ctypes.c_void_p] = {}
_functions: dict[tuple[int, str], ctypes.c_void_p] = {}
_module_locks: dict[tuple[str, int], threading.Lock] = {}
_shared_memory: dict[int, int] = {}


def _function(binary: _Binary, kernel: str, device: int) -> ctypes.c_void_p:
    kit = toolkit()
    lowered = binary.lowered[kernel]
    module_key = (binary.key, device)
    with _memory_lock:
        lock = _module_locks.setdefault(module_key, threading.Lock())
    with lock:
        module = _modules.get(module_key)
        if module is None:
            module = ctypes.c_void_p()
            with torch.cuda.device(device):
                torch.cuda.current_stream(device)  # make the primary context current
                kit.check_driver(
                    kit.ModuleLoadData(ctypes.byref(module), binary.image),
                    "loading runtime-compiled module",
                )
            _modules[module_key] = module
        function_key = (module.value, lowered)
        function = _functions.get(function_key)
        if function is None:
            function = ctypes.c_void_p()
            kit.check_driver(
                kit.ModuleGetFunction(ctypes.byref(function), module, lowered.encode()),
                f"resolving kernel {kernel}",
            )
            _functions[function_key] = function
        return function


def _compiled_parameters(
    binary: _Binary, kernel: str, function: ctypes.c_void_p
) -> tuple[tuple[int, int], ...] | None:
    """``(offset, size)`` of each kernel parameter, or ``None`` if unknown.

    CUDA reports the table through the driver; HIP code objects carry it in
    their AMDGPU metadata note.
    """

    kit = toolkit()
    if kit.hip:
        tables = _amdgpu_tables.get(binary.key)
        if tables is None:
            tables = _amdgpu_tables[binary.key] = amdgpu.kernel_parameters(binary.image)
        return tables.get(binary.lowered[kernel])
    if kit.param_info is None:
        return None
    table = []
    offset, size = ctypes.c_size_t(), ctypes.c_size_t()
    while (
        kit.param_info(function, len(table), ctypes.byref(offset), ctypes.byref(size))
        == 0
    ):
        table.append((offset.value, size.value))
    return tuple(table)


_amdgpu_tables: dict[str, dict[str, amdgpu.ParameterTable]] = {}


def _validate_parameters(
    binary: _Binary,
    function: ctypes.c_void_p,
    launch: CudaLaunch,
    storage: Sequence[Any],
) -> None:
    """Compare packed arguments against the kernel's compiled parameter table.

    Offsets and sizes come from the driver (CUDA) or code-object metadata
    (HIP); pointer element types and scalar kinds come from the mangled name.
    """

    compiled = _compiled_parameters(binary, launch.kernel, function)
    if compiled is not None:
        supplied = []
        offset = 0
        for item in storage:
            alignment = ctypes.alignment(item)
            offset = (offset + alignment - 1) // alignment * alignment
            supplied.append((offset, ctypes.sizeof(item)))
            offset += ctypes.sizeof(item)
        if list(compiled) != supplied:
            raise ValueError(
                f"{launch.kernel}: kernel parameters (offset, size) {list(compiled)} "
                f"do not match launch arguments {supplied}"
            )
    declared = mangling.parameter_types(binary.lowered[launch.kernel])
    if declared is None:
        return
    if len(declared) != len(launch.args):
        raise ValueError(
            f"{launch.kernel}: kernel takes {len(declared)} parameters, "
            f"launch supplies {len(launch.args)}"
        )
    for index, (parameter, argument) in enumerate(zip(declared, launch.args)):
        problem = _type_mismatch(parameter, argument)
        if problem:
            raise ValueError(f"{launch.kernel}: parameter {index} {problem}")


# Kinds of the mangled builtin codes (Itanium ABI) kernels declare.
_BUILTIN_KINDS = {
    **dict.fromkeys(("f", "d", "e", "Dh"), "floating"),
    "b": "bool",
    **dict.fromkeys(("a", "c", "s", "i", "l", "x", "n"), "signed"),
    **dict.fromkeys(("h", "t", "j", "m", "y", "o"), "unsigned"),
}
_CTYPE_KINDS = {
    **{
        native.ctype: _BUILTIN_KINDS[native.mangled[0]]
        for native in ELEMENTS.values()
        if native.ctype is not None
    },
    ctypes.c_void_p: "pointer",
}


def _type_mismatch(parameter: mangling.ParameterType, argument: KernelArgument) -> str:
    supplied = _CTYPE_KINDS.get(argument.ctype)
    if parameter.pointer:
        if supplied != "pointer":
            return f"is a pointer but receives {argument.ctype.__name__}"
        expected = MANGLED_ELEMENTS.get(parameter.builtin)
        if argument.element is not None and expected not in (None, argument.element):
            return f"points to {expected} but receives a {argument.element} tensor"
        return ""
    if parameter.builtin is None:
        return ""
    expected_kind = _BUILTIN_KINDS.get(parameter.builtin)
    if supplied == "pointer" or (
        expected_kind and supplied and supplied != expected_kind
    ):
        return f"is a {expected_kind} value but receives {argument.ctype.__name__}"
    return ""


@dataclass(slots=True)
class _PreparedLaunch:
    function: ctypes.c_void_p
    grid: tuple[int, int, int]
    block: tuple[int, int, int]
    shared_memory: int
    parameters: Any
    storage: list[Any] = field(repr=False)


def prepare(
    request: RtcRequest, steps: Sequence[LaunchStep], device: int
) -> Callable[[], None]:
    """Bind launch steps to their compiled kernels and return one launcher.

    The launcher enqueues on the device's current stream unless it is given an
    explicit raw stream handle.
    """

    binary = compile_request(request, device)
    kit = toolkit()
    prepared: list[_PreparedLaunch | Callable[[], Any]] = []
    for step in steps:
        if not isinstance(step, CudaLaunch):
            prepared.append(step)
            continue
        function = _function(binary, step.kernel, device)
        storage = [argument.storage() for argument in step.args]
        _validate_parameters(binary, function, step, storage)
        parameters = (ctypes.c_void_p * max(1, len(storage)))(
            *(ctypes.cast(ctypes.byref(item), ctypes.c_void_p) for item in storage)
        )
        if step.shared_memory > 48 * 1024:
            with _memory_lock:
                if step.shared_memory > _shared_memory.get(function.value, 0):
                    kit.check_driver(
                        kit.FuncSetAttribute(function, 8, step.shared_memory),
                        f"reserving shared memory for {step.kernel}",
                    )
                    _shared_memory[function.value] = step.shared_memory
        prepared.append(
            _PreparedLaunch(
                function, step.grid, step.block, step.shared_memory, parameters, storage
            )
        )
    launch_kernel = kit.LaunchKernel
    check = kit.check_driver
    current_stream = torch._C._cuda_getCurrentRawStream
    current_device = torch._C._cuda_getDevice
    owners = {}

    def retain(argument):
        if isinstance(argument.owner, torch.Tensor):
            owners[id(argument.owner)] = argument.owner
        if isinstance(argument.value, tuple):
            for child in argument.value:
                retain(child)

    for step in steps:
        if isinstance(step, CudaLaunch):
            for argument in step.args:
                retain(argument)

    # The allocation stays alive through ``owners`` for this binding's entire
    # lifetime. The allocator retains each stream association until that
    # allocation is freed, so recording every launch repeats the same work.
    recorded_streams: dict[int, torch.cuda.Stream] = {}

    def launch(stream: int | None = None) -> None:
        if current_device() != device:
            with torch.cuda.device(device):
                return launch(stream)
        active_stream = current_stream(device)
        if stream is not None and stream != active_stream:
            execution_stream = recorded_streams.get(stream)
            if execution_stream is None:
                execution_stream = torch.cuda.ExternalStream(stream, device=device)
            # Callable steps also enqueue PyTorch operations. They must share
            # the driver's explicit stream and preserve their declared order.
            with torch.cuda.stream(execution_stream):
                return launch()
        stream = active_stream
        if (
            owners
            and stream not in recorded_streams
            and not torch.cuda.is_current_stream_capturing()
        ):
            execution_stream = torch.cuda.ExternalStream(stream, device=device)
            for tensor in owners.values():
                tensor.record_stream(execution_stream)
            recorded_streams[stream] = execution_stream
        for item in prepared:
            if not isinstance(item, _PreparedLaunch):
                item()
                continue
            result = launch_kernel(
                item.function,
                *item.grid,
                *item.block,
                item.shared_memory,
                stream,
                item.parameters,
                None,
            )
            if result:
                check(result, "launching runtime-compiled kernel")

    return launch


def request_for(program: RtcProgram, steps: Sequence[LaunchStep]) -> RtcRequest:
    return RtcRequest(
        program, tuple(step.kernel for step in steps if isinstance(step, CudaLaunch))
    )


# ---------------------------------------------------------------------- #
# Conditional WHILE graphs
# ---------------------------------------------------------------------- #

_CU_GRAPH_NODE_TYPE_CONDITIONAL = 13
_CU_GRAPH_COND_TYPE_WHILE = 1
_CU_GRAPH_COND_ASSIGN_DEFAULT = 1
_CU_STREAM_CAPTURE_MODE_THREAD_LOCAL = 1


class _ConditionalNodeParams(ctypes.Structure):
    """``CUDA_CONDITIONAL_NODE_PARAMS`` of ``cuda.h``."""

    _fields_ = (
        ("handle", ctypes.c_ulonglong),
        ("type", ctypes.c_int),
        ("size", ctypes.c_uint),
        ("phGraph_out", ctypes.POINTER(ctypes.c_void_p)),
        ("ctx", ctypes.c_void_p),
    )


class _NodeParamsUnion(ctypes.Union):
    _fields_ = (
        ("reserved1", ctypes.c_longlong * 29),
        ("conditional", _ConditionalNodeParams),
    )


class _GraphNodeParams(ctypes.Structure):
    """``CUgraphNodeParams`` of ``cuda.h``: 256 bytes."""

    _anonymous_ = ("parameters",)
    _fields_ = (
        ("type", ctypes.c_int),
        ("reserved0", ctypes.c_int * 3),
        ("parameters", _NodeParamsUnion),
        ("reserved2", ctypes.c_longlong),
    )


class _GraphDriver:
    """The driver entry points of conditional graphs (CUDA 12.4 and newer)."""

    def __init__(self) -> None:
        kit = toolkit()
        p, u, i = ctypes.c_void_p, ctypes.c_uint, ctypes.c_int
        pointer = ctypes.POINTER
        for name, symbol, argtypes in (
            ("device", "cuDeviceGet", (pointer(i), i)),
            ("retain", "cuDevicePrimaryCtxRetain", (pointer(p), i)),
            ("release", "cuDevicePrimaryCtxRelease_v2", (i,)),
            ("create", "cuGraphCreate", (pointer(p), u)),
            (
                "handle",
                "cuGraphConditionalHandleCreate",
                (pointer(ctypes.c_ulonglong), p, p, u, u),
            ),
            (
                "add_node",
                "cuGraphAddNode_v2",
                (pointer(p), p, p, p, ctypes.c_size_t, pointer(_GraphNodeParams)),
            ),
            ("begin", "cuStreamBeginCapture_v2", (p, i)),
            ("end", "cuStreamEndCapture", (p, pointer(p))),
            (
                "add_child",
                "cuGraphAddChildGraphNode",
                (pointer(p), p, p, ctypes.c_size_t, p),
            ),
            (
                "instantiate",
                "cuGraphInstantiateWithFlags",
                (pointer(p), p, ctypes.c_ulonglong),
            ),
            ("launch", "cuGraphLaunch", (p, p)),
            ("destroy_exec", "cuGraphExecDestroy", (p,)),
            ("destroy", "cuGraphDestroy", (p,)),
        ):
            setattr(self, name, _bind(kit.driver, symbol, *argtypes))
        self.check = kit.check_driver


@cache
def _graph_driver() -> _GraphDriver:
    return _GraphDriver()


@cache
def _conditional_graph_support() -> bool:
    try:
        kit = toolkit()
        if kit.hip or kit.version < (12, 4) or kit.driver_version < 12040:
            return False
        _graph_driver()
        # The WHILE condition kernel includes the device-runtime declarations.
        return bool(toolkit_include_options())
    except (OSError, AttributeError, RuntimeError):
        return False


def conditional_graphs(device: torch.device) -> bool:
    """Whether HydroForge's CUDA conditional WHILE graphs run on ``device``.

    PyTorch exposes AMD/ROCm devices through the ``cuda`` device type, but HIP
    has no conditional graph nodes. CUDA needs a driver and runtime compiler
    of 12.4 or newer and the toolkit's device-runtime header.
    """

    return device.type == "cuda" and _conditional_graph_support()


class ConditionalWhileGraph:
    """Owns one CUDA conditional-graph ``WHILE`` node and its instantiation.

    The loop body is captured into the node's body graph by
    :meth:`begin_capture` / :meth:`end_capture`; the body sets the condition
    through :attr:`handle`.  :meth:`launch` then runs the whole loop from a
    single host launch, and the body runs at least once per launch.
    """

    def __init__(self) -> None:
        api = _graph_driver()
        self._device = torch.cuda.current_device()
        self._graph = self._exec = None
        # The primary context is the one PyTorch launches into; retaining it
        # keeps the conditional handle's context alive for this graph's life.
        cu_device = ctypes.c_int()
        api.check(api.device(ctypes.byref(cu_device), self._device), "device lookup")
        context = ctypes.c_void_p()
        api.check(
            api.retain(ctypes.byref(context), cu_device), "primary context retain"
        )
        self._cu_device, self._context = cu_device, context
        try:
            graph = ctypes.c_void_p()
            api.check(api.create(ctypes.byref(graph), 0), "graph creation")
            self._graph = graph
            # Default value 1 makes the body run at least once per launch.
            handle = ctypes.c_ulonglong()
            api.check(
                api.handle(
                    ctypes.byref(handle),
                    graph,
                    context,
                    1,
                    _CU_GRAPH_COND_ASSIGN_DEFAULT,
                ),
                "conditional handle creation",
            )
            self._handle = handle.value
            params = _GraphNodeParams(type=_CU_GRAPH_NODE_TYPE_CONDITIONAL)
            params.conditional.handle = self._handle
            params.conditional.type = _CU_GRAPH_COND_TYPE_WHILE
            params.conditional.size = 1
            params.conditional.ctx = context.value
            node = ctypes.c_void_p()
            api.check(
                api.add_node(
                    ctypes.byref(node), graph, None, None, 0, ctypes.byref(params)
                ),
                "conditional node creation",
            )
            self._body = ctypes.c_void_p(params.conditional.phGraph_out[0])
        except BaseException:
            with cleanup_on_exit("conditional graph construction", (self.destroy,)):
                raise

    @property
    def handle(self) -> int:
        """The conditional handle a body control kernel sets."""
        return self._handle

    def begin_capture(self, stream_ptr: int) -> None:
        api = _graph_driver()
        with torch.cuda.device(self._device):
            api.check(
                api.begin(stream_ptr, _CU_STREAM_CAPTURE_MODE_THREAD_LOCAL),
                "stream capture begin",
            )

    def end_capture(self, stream_ptr: int) -> None:
        api = _graph_driver()
        with torch.cuda.device(self._device):
            captured = ctypes.c_void_p()
            api.check(api.end(stream_ptr, ctypes.byref(captured)), "stream capture end")
            with cleanup_on_exit(
                "captured graph insertion",
                (
                    lambda: api.check(
                        api.destroy(captured), "captured graph destruction"
                    ),
                ),
            ):
                node = ctypes.c_void_p()
                api.check(
                    api.add_child(ctypes.byref(node), self._body, None, 0, captured),
                    "loop body insertion",
                )

    def instantiate(self) -> None:
        api = _graph_driver()
        with torch.cuda.device(self._device):
            executable = ctypes.c_void_p()
            api.check(
                api.instantiate(ctypes.byref(executable), self._graph, 0),
                "graph instantiation",
            )
            self._exec = executable

    def launch(self) -> None:
        """Launch the loop on the current stream of the graph's device."""

        api = _graph_driver()
        stream = torch._C._cuda_getCurrentRawStream(self._device)
        result = api.launch(self._exec, stream)
        if result:
            api.check(result, "graph launch")

    def destroy(self) -> None:
        context = getattr(self, "_context", None)
        if context is None:
            return
        api = _graph_driver()
        executable, graph = self._exec, self._graph
        self._exec = self._graph = self._context = None
        with cleanup_on_exit(
            "conditional graph",
            (
                *(
                    (
                        lambda: api.check(
                            api.destroy_exec(executable), "graph exec destruction"
                        ),
                    )
                    if executable is not None
                    else ()
                ),
                *(
                    (lambda: api.check(api.destroy(graph), "graph destruction"),)
                    if graph is not None
                    else ()
                ),
                lambda: api.check(
                    api.release(self._cu_device), "primary context release"
                ),
            ),
        ):
            pass

    def __del__(self) -> None:
        try:
            self.destroy()
        except Exception:
            pass


__all__ = [
    "ConditionalWhileGraph",
    "CudaLaunch",
    "KernelArgument",
    "LaunchStep",
    "RTC_PRELUDE",
    "RtcProgram",
    "RtcRequest",
    "blocks",
    "boolean",
    "compile_request",
    "conditional_graphs",
    "ctype",
    "float32",
    "float64",
    "int32",
    "int64",
    "pointer",
    "precompile_request",
    "prepare",
    "program_options",
    "request_for",
    "scalar",
    "struct",
    "toolkit_include_options",
    "uint32",
    "uint64",
]

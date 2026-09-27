"""Runtime compilation and launch of CUDA/HIP device code.

NVRTC (hiprtc under ROCm) compiles device-only sources in process, so no host
compiler, CPython binding or ATen headers are parsed. A program compiles only
the kernel instantiations its launch plans name; the binary is cached on disk by
content. Launches pass pre-packed arguments to the driver on the caller's
current stream, which keeps them valid under CUDA Graph capture.
"""

from __future__ import annotations

import ctypes
import hashlib
import json
import os
import sys
import threading
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from functools import cache
from pathlib import Path
from typing import Any

import torch

from hydroforge.kernels.backends.compile_lock import (
    acquire_compile_lock,
    release_compile_lock,
)
from hydroforge.kernels.backends.cuda import amdgpu, mangling
from hydroforge.serialization.files import atomic_output_path, atomic_write_text

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

_CACHE_VERSION = 1


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

    def _bind_rtc(self, symbol: str, *argtypes: Any, alias: str | None = None) -> None:
        function = getattr(self.rtc, f"{self._rtc_prefix}{symbol}")
        function.argtypes, function.restype = argtypes, ctypes.c_int
        setattr(self, alias or symbol, function)

    def _bind_driver(
        self, symbol: str, *argtypes: Any, alias: str | None = None
    ) -> None:
        function = getattr(self.driver, f"{self._driver_prefix}{symbol}")
        function.argtypes, function.restype = argtypes, ctypes.c_int
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
    image: bytes
    lowered: dict[str, str]


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
    number = getattr(kit.rtc, "nvrtcGetNumSupportedArchs")
    number.argtypes, number.restype = (ctypes.POINTER(ctypes.c_int),), ctypes.c_int
    kit.check_rtc(number(ctypes.byref(count)))
    values = (ctypes.c_int * count.value)()
    listing = getattr(kit.rtc, "nvrtcGetSupportedArchs")
    listing.argtypes, listing.restype = (ctypes.POINTER(ctypes.c_int),), ctypes.c_int
    kit.check_rtc(listing(values))
    return frozenset(values)


def _key(request: RtcRequest, target: str) -> str:
    kit = toolkit()
    identity = {
        "version": _CACHE_VERSION,
        "source": request.program.source,
        "options": list(request.program.options),
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


def _compile(request: RtcRequest, target: str, device_binary: bool) -> _Binary:
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
    try:
        for kernel in request.kernels:
            kit.check_rtc(kit.AddNameExpression(program, kernel.encode()))
        arch = f"--offload-arch={target}" if kit.hip else f"--gpu-architecture={target}"
        options = [arch, "--std=c++20", *_include_options(), *_options(request.program)]
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
        return _Binary(image.raw, lowered)
    finally:
        kit.check_rtc(kit.DestroyProgram(ctypes.byref(program)))


def _program_log(program: ctypes.c_void_p) -> str:
    kit = toolkit()
    size = ctypes.c_size_t()
    kit.check_rtc(kit.GetProgramLogSize(program, ctypes.byref(size)))
    log = ctypes.create_string_buffer(size.value)
    kit.check_rtc(kit.GetProgramLog(program, log))
    return log.value.decode(errors="replace")


# hiprtc spells the numerical controls as Clang flags; ptxas has no AMD analog.
_HIP_OPTIONS = {
    "--use_fast_math": "-ffast-math",
    "--ftz=false": "-fno-gpu-flush-denormals-to-zero",
    "--ftz=true": "-fgpu-flush-denormals-to-zero",
    "--fmad=false": "-ffp-contract=off",
    "--fmad=true": "-ffp-contract=fast",
}


def _options(program: RtcProgram) -> tuple[str, ...]:
    """Program options in the active runtime compiler's spelling."""
    if not toolkit().hip:
        return program.options
    return tuple(
        _HIP_OPTIONS.get(option, option)
        for option in program.options
        if not option.startswith("--ptxas-options")
    )


@cache
def _include_options() -> tuple[str, ...]:
    """Expose libcu++ and the CUDA headers shipped with the toolkit."""
    if toolkit().hip:
        return ()
    from torch.utils.cpp_extension import include_paths

    roots = [Path(path) for path in include_paths("cuda")]
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
    from torch.utils.cpp_extension import include_paths

    return tuple(
        f"-I{path}"
        for path in dict.fromkeys(Path(p) for p in include_paths("cuda"))
        if (path / "cuda_device_runtime_api.h").is_file()
    )


_memory: dict[str, _Binary] = {}
_memory_lock = threading.Lock()


def _force_rebuild() -> bool:
    value = os.environ.get("HYDROFORGE_CUDA_REBUILD", "")
    return value.strip().lower() in {"1", "true", "yes", "on"}


def compile_request(request: RtcRequest, device: int) -> _Binary:
    """Return the cached binary for ``request``, compiling it at most once."""

    target, device_binary = _target(device)
    key = _key(request, target)
    with _memory_lock:
        cached = _memory.get(key)
    if cached is not None:
        return cached
    directory = _cache_root() / key[:2]
    image_path = directory / f"{key}.bin"
    names_path = directory / f"{key}.json"

    def load() -> _Binary | None:
        if _force_rebuild() or not (image_path.is_file() and names_path.is_file()):
            return None
        return _Binary(image_path.read_bytes(), json.loads(names_path.read_text()))

    binary = load()
    if binary is None:
        directory.mkdir(parents=True, exist_ok=True)
        lock = directory / f"{key}.lock"
        binary = acquire_compile_lock(
            lock,
            env_prefix="HYDROFORGE",
            verbose=False,
            cache_probe=load,
        )
        if binary is None:
            try:
                binary = _compile(request, target, device_binary)
                with atomic_output_path(image_path) as temporary:
                    temporary.write_bytes(binary.image)
                atomic_write_text(names_path, json.dumps(binary.lowered))
            finally:
                release_compile_lock(lock)
    with _memory_lock:
        _memory.setdefault(key, binary)
    return binary


def compile_requests(requests: Iterable[RtcRequest], device: int, *, jobs: int) -> None:
    """Compile distinct requests concurrently; the runtime compiler releases the GIL."""

    unique = list(dict.fromkeys(requests))
    if not unique:
        return
    # Largest programs first: the slowest compile then starts in the first
    # wave instead of lengthening the tail (cached requests return at once).
    unique.sort(
        key=lambda request: len(request.program.source) * len(request.kernels),
        reverse=True,
    )
    if jobs == 1 or len(unique) == 1:
        for request in unique:
            compile_request(request, device)
        return
    with ThreadPoolExecutor(max_workers=min(jobs, len(unique))) as pool:
        for future in [pool.submit(compile_request, r, device) for r in unique]:
            future.result()


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
    return KernelArgument(ctypes.c_void_p, tensor.data_ptr(), tensor.dtype)


def boolean(value: bool) -> KernelArgument:
    return KernelArgument(ctypes.c_bool, bool(value))


def int32(value: int) -> KernelArgument:
    return KernelArgument(ctypes.c_int32, _checked_int(value, 32))


def uint32(value: int) -> KernelArgument:
    if not 0 <= value < 2**32:
        raise ValueError(f"uint32 kernel argument out of range: {value}")
    return KernelArgument(ctypes.c_uint32, value)


def uint64(value: int) -> KernelArgument:
    if not 0 <= value < 2**64:
        raise ValueError(f"uint64 kernel argument out of range: {value}")
    return KernelArgument(ctypes.c_uint64, value)


def int64(value: int) -> KernelArgument:
    return KernelArgument(ctypes.c_int64, _checked_int(value, 64))


def float32(value: float) -> KernelArgument:
    return KernelArgument(ctypes.c_float, value)


def float64(value: float) -> KernelArgument:
    return KernelArgument(ctypes.c_double, value)


def scalar(value: Any, dtype: torch.dtype) -> KernelArgument:
    """Scalar in the device type of ``dtype`` (e.g. a precision-typed ``Real``)."""

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


_SCALAR_ARGUMENTS = {
    torch.float32: float32,
    torch.float64: float64,
    torch.int32: int32,
    torch.int64: int64,
    torch.bool: boolean,
}
_CTYPE_NAMES = {
    torch.float32: "float",
    torch.float64: "double",
    torch.int32: "int32_t",
    torch.int64: "int64_t",
    torch.bool: "bool",
    torch.uint8: "uint8_t",
    torch.int8: "int8_t",
}


def ctype(value: torch.Tensor | torch.dtype) -> str:
    """C++ spelling of a tensor's element type for template arguments."""

    dtype = value.dtype if isinstance(value, torch.Tensor) else value
    try:
        return _CTYPE_NAMES[dtype]
    except KeyError:
        raise TypeError(f"no device element type for {dtype}") from None


def _dim3(value: int | Sequence[int], what: str) -> tuple[int, int, int]:
    values = (value,) if isinstance(value, int) else tuple(value)
    if not 1 <= len(values) <= 3 or any(
        isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values
    ):
        raise ValueError(f"CUDA {what} must be 1-3 positive ints, got {value!r}")
    if what == "grid" and (values[0] >= 2**31 or any(v > 65535 for v in values[1:])):
        raise ValueError(f"CUDA grid exceeds device limits: {values}")
    if what == "block" and prod_int(values) > 1024:
        raise ValueError(f"CUDA block exceeds 1024 threads: {values}")
    return (*values, *(1,) * (3 - len(values)))


def prod_int(values: Iterable[int]) -> int:
    result = 1
    for value in values:
        result *= value
    return result


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
        if self.shared_memory < 0:
            raise ValueError("dynamic shared memory must be nonnegative")


LaunchStep = CudaLaunch | Callable[[], Any]


def blocks(count: int, threads: int) -> int:
    """Number of ``threads``-sized blocks covering ``count`` items."""

    return (count - 1) // threads + 1


# ---------------------------------------------------------------------- #
# Loading and prepared launches
# ---------------------------------------------------------------------- #

_modules: dict[tuple[str, int], ctypes.c_void_p] = {}
_functions: dict[tuple[int, str], ctypes.c_void_p] = {}


def _function(binary: _Binary, kernel: str, device: int) -> ctypes.c_void_p:
    kit = toolkit()
    lowered = binary.lowered[kernel]
    module_key = (hashlib.sha256(binary.image).hexdigest(), device)
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
        digest = hashlib.sha256(binary.image).hexdigest()
        tables = _amdgpu_tables.get(digest)
        if tables is None:
            tables = _amdgpu_tables[digest] = amdgpu.kernel_parameters(binary.image)
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


# Mangled builtin codes (Itanium ABI) of the element and scalar types kernels use.
_ELEMENT_DTYPES = {
    "f": torch.float32, "d": torch.float64, "Dh": torch.float16, "b": torch.bool,
    "a": torch.int8, "c": torch.int8, "h": torch.uint8, "s": torch.int16,
    "t": torch.uint16, "i": torch.int32, "j": torch.uint32, "l": torch.int64,
    "x": torch.int64, "m": torch.uint64, "y": torch.uint64,
}  # fmt: skip
_BUILTIN_KINDS = {
    **dict.fromkeys(("f", "d", "e", "Dh"), "floating"),
    "b": "bool",
    **dict.fromkeys(("a", "c", "s", "i", "l", "x", "n"), "signed"),
    **dict.fromkeys(("h", "t", "j", "m", "y", "o"), "unsigned"),
}
_CTYPE_KINDS = {
    ctypes.c_float: "floating", ctypes.c_double: "floating", ctypes.c_bool: "bool",
    ctypes.c_int8: "signed", ctypes.c_int16: "signed", ctypes.c_int32: "signed",
    ctypes.c_int64: "signed", ctypes.c_uint8: "unsigned", ctypes.c_uint16: "unsigned",
    ctypes.c_uint32: "unsigned", ctypes.c_uint64: "unsigned",
    ctypes.c_void_p: "pointer",
}  # fmt: skip


def _type_mismatch(parameter: mangling.ParameterType, argument: KernelArgument) -> str:
    supplied = _CTYPE_KINDS.get(argument.ctype)
    if parameter.pointer:
        if supplied != "pointer":
            return f"is a pointer but receives {argument.ctype.__name__}"
        expected = _ELEMENT_DTYPES.get(parameter.builtin)
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
            kit.check_driver(
                kit.FuncSetAttribute(function, 8, step.shared_memory),
                f"reserving shared memory for {step.kernel}",
            )
        prepared.append(
            _PreparedLaunch(
                function, step.grid, step.block, step.shared_memory, parameters, storage
            )
        )
    launch_kernel = kit.LaunchKernel
    check = kit.check_driver
    current_stream = torch._C._cuda_getCurrentRawStream
    current_device = torch._C._cuda_getDevice

    def launch(stream: int | None = None) -> None:
        if current_device() != device:
            with torch.cuda.device(device):
                return launch(stream)
        if stream is None:
            stream = current_stream(device)
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


__all__ = [
    "CudaLaunch",
    "KernelArgument",
    "LaunchStep",
    "RTC_PRELUDE",
    "RtcProgram",
    "RtcRequest",
    "blocks",
    "boolean",
    "compile_request",
    "compile_requests",
    "ctype",
    "float32",
    "float64",
    "int32",
    "int64",
    "pointer",
    "prepare",
    "request_for",
    "scalar",
    "struct",
    "toolkit_include_options",
    "uint32",
    "uint64",
]

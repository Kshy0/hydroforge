"""Demand-driven scalar time aggregation, lowered to each execution backend."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING
from uuid import uuid4

import torch

from hydroforge.compiler.generated import (
    compile_generated_module,
    release_generated_module,
)
from hydroforge.statistics.ir import Expression, ExpressionDialect, render_expression

if TYPE_CHECKING:
    from hydroforge.execution.capture import CaptureRuntime

DAY_MICROSECONDS = 86_400_000_000


@dataclass(frozen=True)
class StepFieldOutput:
    source: str
    dtype: torch.dtype
    slot: int


@dataclass(frozen=True)
class StepFieldCompileContext:
    calendar: str
    has_year_zero: bool
    outputs: tuple[StepFieldOutput, ...]
    expressions: Mapping[str, Expression]

    def dependencies(self) -> tuple[str, ...]:
        ordered: dict[str, None] = {}

        def visit(name: str) -> None:
            if name in ordered:
                return
            expression = self.expressions.get(name)
            if expression is not None:
                for dependency in expression.dependencies:
                    visit(dependency)
            ordered[name] = None

        for output in self.outputs:
            visit(output.source)
        return tuple(ordered)

    @property
    def requires_calendar(self) -> bool:
        return any(
            source not in self.expressions and source != "step_seconds"
            for source in self.dependencies()
        )


class _TimeEmitter:
    def __init__(
        self, context: StepFieldCompileContext, dialect: ExpressionDialect
    ) -> None:
        self.context = context
        self.dialect = dialect
        self.python = dialect in {ExpressionDialect.TORCH, ExpressionDialect.TRITON}
        self.floating = "float32" if dialect is ExpressionDialect.METAL else "float64"
        self.lines: list[str] = []
        self.indent = 1
        self.declared: set[str] = set()

    def append(self, line: str) -> None:
        self.lines.append("    " * self.indent + line)

    def cast(self, value: str, dtype: str) -> str:
        if self.dialect is ExpressionDialect.TRITON:
            return f"tl.cast({value}, tl.{dtype})"
        if self.dialect is ExpressionDialect.TORCH:
            return f"torch.as_tensor({value}, dtype=torch.{dtype}, device=clock.device)"
        native = {
            "int64": "long" if self.dialect is ExpressionDialect.METAL else "int64_t",
            "float32": "float",
            "float64": "double",
        }[dtype]
        return f"{native}({value})"

    def assign(self, name: str, value: str, dtype: str = "int64") -> None:
        prefix = "auto " if not self.python and name not in self.declared else ""
        suffix = "" if self.python else ";"
        self.append(f"{prefix}{name} = {self.cast(value, dtype)}{suffix}")
        self.declared.add(name)

    def begin(self, kind: str, condition: str) -> None:
        self.append(
            f"{kind} {condition}:" if self.python else f"{kind} ({condition}) {{"
        )
        self.indent += 1

    def end(self) -> None:
        self.indent -= 1
        if not self.python:
            self.append("}")

    def select(self, condition: str, positive: str, negative: str) -> str:
        if self.dialect is ExpressionDialect.TRITON:
            return f"tl.where({condition}, {positive}, {negative})"
        if self.dialect is ExpressionDialect.TORCH:
            return f"torch.where({condition}, {positive}, {negative})"
        return f"(({condition}) ? ({positive}) : ({negative}))"

    def load(self, buffer: str, index: int) -> str:
        if self.dialect is ExpressionDialect.TRITON:
            return f"tl.load({buffer} + {index})"
        prefix = "args." if self.dialect is ExpressionDialect.METAL else ""
        return f"{prefix}{buffer}[{index}]"

    def store(self, buffer: str, index: int, value: str) -> None:
        if self.dialect is ExpressionDialect.TRITON:
            self.append(f"tl.store({buffer} + {index}, {value})")
        else:
            suffix = "" if self.python else ";"
            self.append(f"{self.load(buffer, index)} = {value}{suffix}")

    def leap(self) -> str:
        calendar = self.context.calendar
        if calendar == "all_leap":
            return "1"
        if calendar in {"noleap", "360_day"}:
            return "0"
        julian = "(clock_year % 4 == 0)"
        gregorian = "((clock_year % 4 == 0) & ((clock_year % 100 != 0) | (clock_year % 400 == 0)))"
        if calendar == "julian":
            return julian
        if calendar == "standard":
            return self.select("clock_year < 1582", julian, gregorian)
        return gregorian

    def year_length(self) -> str:
        if self.context.calendar == "360_day":
            return "360"
        result = f"365 + {self.cast(self.leap(), 'int64')}"
        if self.context.calendar == "standard":
            result += f" - {self.select('clock_year == 1582', '10', '0')}"
        return result

    def body(self) -> str:
        dependencies = self.context.dependencies()
        self.assign("duration_days", self.load("duration", 0))
        self.assign("duration_micros", self.load("duration", 1))
        if self.context.requires_calendar:
            self.assign("clock_year", self.load("clock", 0))
            self.assign("ordinal", self.load("clock", 1))
            self.assign("micros", self.load("clock", 2))
            self.assign("year_length", self.year_length())
            condition = (
                "*args.advance"
                if self.dialect is ExpressionDialect.METAL
                else "advance"
            )
            self.begin("if", condition)
            self.assign("micros", f"micros + {self.load('clock', 4)}")
            division = "//" if self.python else "/"
            self.assign(
                "ordinal",
                f"ordinal + {self.load('clock', 3)} + micros {division} {DAY_MICROSECONDS}",
            )
            self.assign("micros", f"micros % {DAY_MICROSECONDS}")
            cycle = {
                "proleptic_gregorian": (400, 146097),
                "julian": (4, 1461),
                "noleap": (1, 365),
                "all_leap": (1, 366),
                "360_day": (1, 360),
            }.get(self.context.calendar)
            if cycle is not None:
                cycle_years, cycle_days = cycle
                self.assign("cycles", f"ordinal {division} {cycle_days}")
                self.assign("ordinal", f"ordinal - cycles * {cycle_days}")
                self.assign("clock_year", f"clock_year + cycles * {cycle_years}")
            self.assign("year_length", self.year_length())
            self.begin("while", "ordinal >= year_length")
            self.assign("ordinal", "ordinal - year_length")
            self.assign("clock_year", "clock_year + 1")
            self.assign("year_length", self.year_length())
            self.end()
            self.end()
            self.assign("year_length", self.year_length())
            self.assign("nominal_day", "ordinal")
            if self.context.calendar == "standard":
                self.assign(
                    "nominal_day",
                    "ordinal + "
                    + self.select("(clock_year == 1582) & (ordinal >= 277)", "10", "0"),
                )
            self.assign("leap_day", self.leap())
            self.assign("calendar_month", "0")
            self.assign("month_start", "0")
            for month, offset in enumerate(
                (31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334), 1
            ):
                start = (
                    str(month * 30)
                    if self.context.calendar == "360_day"
                    else str(offset)
                    if month == 1
                    else f"({offset} + leap_day)"
                )
                condition = f"nominal_day >= {start}"
                self.assign(
                    "calendar_month",
                    self.select(condition, str(month), "calendar_month"),
                )
                self.assign("month_start", self.select(condition, start, "month_start"))
            for index, value in enumerate(
                ("clock_year", "ordinal", "micros", "duration_days", "duration_micros")
            ):
                self.store("clock", index, value)
        values = {
            "doy": ("nominal_day + 1", "int64"),
            "month_idx": ("calendar_month", "int64"),
            "days_in_year": ("year_length", "int64"),
            "day_seconds": (
                f"{self.cast('micros', self.floating)} / 1000000.0",
                self.floating,
            ),
            "julian": (
                f"{self.cast('nominal_day', self.floating)} + {self.cast('micros', self.floating)} / 1000000.0 / 86400.0",
                self.floating,
            ),
            "step_seconds": (
                f"{self.cast('duration_days', self.floating)} * 86400.0 + {self.cast('duration_micros', self.floating)} / 1000000.0",
                self.floating,
            ),
            "year": (
                "clock_year"
                if self.context.has_year_zero
                else "clock_year - " + self.select("clock_year <= 0", "1", "0"),
                "int64",
            ),
            "month": ("calendar_month + 1", "int64"),
            "day": ("nominal_day - month_start + 1", "int64"),
        }
        names = {}
        for index, source in enumerate(dependencies):
            symbol = f"field_{index}"
            if source in self.context.expressions:
                expression = render_expression(
                    self.context.expressions[source],
                    self.dialect,
                    names,
                    value_type=self.floating,
                )
                self.assign(symbol, expression, self.floating)
            else:
                expression, dtype = values[source]
                self.assign(symbol, expression, dtype)
            names[source] = symbol
        dtypes = tuple(dict.fromkeys(output.dtype for output in self.context.outputs))
        for output in self.context.outputs:
            self.store(
                f"slab_{dtypes.index(output.dtype)}", output.slot, names[output.source]
            )
        return "\n".join(self.lines)


def generate_step_field_source(
    context: StepFieldCompileContext, dialect: ExpressionDialect
) -> str:
    """Emit one scalar aggregator; advancing and refresh share the same body."""
    return _TimeEmitter(context, dialect).body()


class CompiledStepFields:
    """Execution-owned generated kernel and optional single-node replay graph."""

    def __init__(
        self,
        context: StepFieldCompileContext,
        *,
        backend: str,
        clock: torch.Tensor,
        duration: torch.Tensor,
        storage: Mapping[torch.dtype, torch.Tensor],
        capture: CaptureRuntime | None = None,
    ) -> None:
        self.context = context
        self.clock = clock
        self.duration = duration
        self.dtypes = tuple(dict.fromkeys(output.dtype for output in context.outputs))
        self.tensors = (clock, duration, *(storage[dtype] for dtype in self.dtypes))
        self.capture = capture
        self.graph = None
        self.module = None
        self.filename = None
        if clock.device.type == "cpu":
            dialect = ExpressionDialect.TORCH
        elif backend == "triton" or clock.device.type == "xpu":
            dialect = ExpressionDialect.TRITON
        elif clock.device.type == "mps":
            dialect = ExpressionDialect.METAL
        elif clock.device.type == "cuda":
            dialect = ExpressionDialect.CUDA
        else:
            raise ValueError(
                f"compiled step fields do not support device {clock.device}"
            )
        self.dialect = dialect
        self.source = generate_step_field_source(context, dialect)
        if dialect in {ExpressionDialect.TORCH, ExpressionDialect.TRITON}:
            self._compile_python()
        elif dialect is ExpressionDialect.CUDA:
            self._compile_cuda()
        else:
            self._compile_metal()

    def _compile_python(self) -> None:
        parameters = [
            "clock",
            "duration",
            *(f"slab_{index}" for index in range(len(self.dtypes))),
        ]
        if self.dialect is ExpressionDialect.TRITON:
            header = (
                "import triton\nimport triton.language as tl\nfrom triton.language.extra import libdevice\n"
                "@triton.jit\ndef hydroforge_maximum(left, right):\n"
                "    return tl.where(left != left, right, tl.where(right != right, left, tl.maximum(left, right)))\n"
                "@triton.jit\ndef hydroforge_minimum(left, right):\n"
                "    return tl.where(left != left, right, tl.where(right != right, left, tl.minimum(left, right)))\n"
                "@triton.jit\n"
            )
            parameters.append("advance: tl.constexpr")
        else:
            header = (
                "import torch\n"
                "hydroforge_where = torch.where\n"
                "hydroforge_maximum = torch.fmax\n"
                "hydroforge_minimum = torch.fmin\n"
                "hydroforge_remainder = torch.remainder\n"
            )
            parameters.append("advance")
        source = (
            header
            + f"def aggregate_time({', '.join(parameters)}):\n"
            + self.source
            + "\n"
        )
        name = "hydroforge_step_time_" + uuid4().hex
        module = compile_generated_module(source, name=name)
        self.module = module
        self.filename = module.__file__
        self.source = source
        kernel = module.aggregate_time
        if self.dialect is ExpressionDialect.TRITON:
            self.launch = lambda advance: kernel[(1,)](
                *self.tensors, advance, num_warps=1, enable_fp_fusion=False
            )
        else:
            self.launch = lambda advance: kernel(*self.tensors, advance)

    def _compile_cuda(self) -> None:
        from hydroforge.kernels.backends.cuda.build import load_inline_cu_module

        types = {
            torch.int32: "int32_t",
            torch.int64: "int64_t",
            torch.float32: "float",
            torch.float64: "double",
        }
        parameters = ["int64_t* clock", "const int64_t* duration"]
        parameters.extend(
            f"{types[dtype]}* slab_{index}" for index, dtype in enumerate(self.dtypes)
        )
        parameters.append("bool advance")
        declaration = (
            "void aggregate_time(std::vector<at::Tensor> tensors, bool advance);"
        )
        arguments = ["tensors[0].data_ptr<int64_t>()", "tensors[1].data_ptr<int64_t>()"]
        arguments.extend(
            f"tensors[{index + 2}].data_ptr<{types[dtype]}>()"
            for index, dtype in enumerate(self.dtypes)
        )
        source = (
            "#include <torch/extension.h>\n#include <c10/cuda/CUDAStream.h>\n"
            "#include <c10/cuda/CUDAGuard.h>\n#include <c10/cuda/CUDAException.h>\n"
            "#include <cmath>\n"
            f"__global__ void time_kernel({', '.join(parameters)}) {{\n{self.source.replace('hf_max(', 'fmax(').replace('hf_min(', 'fmin(')}\n}}\n"
            "void aggregate_time(std::vector<at::Tensor> tensors, bool advance) {\n"
            "    const c10::cuda::CUDAGuard guard(tensors[0].device());\n"
            f"    time_kernel<<<1, 1, 0, c10::cuda::getCurrentCUDAStream()>>>({', '.join(arguments)}, advance);\n"
            "    C10_CUDA_KERNEL_LAUNCH_CHECK();\n}\n"
        )
        digest = hashlib.sha256(source.encode()).hexdigest()[:16]
        self.module = load_inline_cu_module(
            "hydroforge_step_time_" + digest,
            cpp_sources=declaration,
            cuda_sources=source,
            functions=["aggregate_time"],
            extra_cuda_cflags=(
                "-O3",
                "-ffp-contract=off" if torch.version.hip else "--fmad=false",
            ),
        )
        self.source = source
        self.launch = lambda advance: self.module.aggregate_time(
            list(self.tensors), advance
        )

    def _compile_metal(self) -> None:
        from hydroforge.kernels.backends.metal.online import (
            MetalBuffer,
            MetalScalar,
            make_online_metal_dispatcher,
        )

        buffers = (
            MetalBuffer("clock", torch.int64, "read_write"),
            MetalBuffer("duration", torch.int64, "read"),
        )
        buffers += tuple(
            MetalBuffer(f"slab_{index}", dtype, "write")
            for index, dtype in enumerate(self.dtypes)
        )
        dispatcher = make_online_metal_dispatcher(
            "hf_aggregate_time",
            buffers=buffers,
            scalars=(MetalScalar("advance", "bool"), MetalScalar("count", "index")),
            size_key="count",
            body="if (i == 0) {\n"
            + self.source.replace("hydroforge_maximum(", "fmax(").replace(
                "hydroforge_minimum(", "fmin("
            )
            + "\n}",
        )
        launches = {}
        for advance in (False, True):
            arguments = dict(
                zip((buffer.name for buffer in buffers), self.tensors, strict=True)
            )
            arguments.update(advance=advance, count=1, BLOCK_SIZE=1)
            dtypes = {buffer.name: buffer.dtype for buffer in buffers}
            dispatcher._validate_specialization_input(arguments, buffer_dtypes=dtypes)
            launches[advance] = dispatcher.specialize(arguments, buffer_dtypes=dtypes)
        self.launch = lambda advance: launches[advance]()

    def run(self, *, advance: bool) -> None:
        if advance and self.capture is not None:
            if self.graph is None:
                self.graph = self.capture.capture_cuda(
                    lambda: self.launch(True),
                    mutated_state=self.tensors,
                )
            self.graph.replay()
        else:
            self.launch(advance)

    def close(self) -> None:
        try:
            if self.graph is not None:
                graph, self.graph = self.graph, None
                self.capture.release(graph)
        finally:
            if self.filename is not None:
                release_generated_module(self.module.__name__, self.filename)
                self.filename = None
            self.module = None
            self.launch = None
            self.tensors = ()

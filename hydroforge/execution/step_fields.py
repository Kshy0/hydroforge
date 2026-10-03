"""Address-stable time bindings driven by a compiled device clock aggregator."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, replace
from datetime import timedelta
from functools import partial
from typing import TYPE_CHECKING, Any
from uuid import uuid4

import torch
from torch.utils._python_dispatch import _disable_current_modes

from hydroforge.compiler.step_fields import (
    StepFieldCompileContext,
    StepFieldOutput,
    step_field_program,
)
from hydroforge.contracts.step_fields import (
    _BUILTIN_STEP_FIELDS,
    StepField,
    StepTime,
    _StepFieldValues,
)
from hydroforge.core.errors import cleanup_on_exit
from hydroforge.execution.aten import normalize_floating_scalar
from hydroforge.kernels.calls import recording_sink
from hydroforge.kernels.codegen.c import CUDA, MSL, msl_arguments
from hydroforge.kernels.codegen.ir import KernelFunction
from hydroforge.kernels.codegen.torch import PRELUDE as TORCH_PRELUDE
from hydroforge.kernels.codegen.torch import TorchPrinter
from hydroforge.kernels.codegen.triton import PRINTER as TRITON
from hydroforge.kernels.toolchain import CompileRequest
from hydroforge.kernels.toolchain.python import (
    compile_generated_module,
    release_generated_module,
)
from hydroforge.platform.backend import Backend

if TYPE_CHECKING:
    from hydroforge.execution.executors import LoopExecutor


def _upload(destination: torch.Tensor, source: torch.Tensor) -> None:
    """Stream-order a host update without waiting for queued device work."""

    if destination.device.type == "cuda":
        # Caching-host-allocator staging is not reused until this copy has
        # completed, so later host edits of ``source`` cannot race it.
        destination.copy_(source.pin_memory(), non_blocking=True)
    else:
        destination.copy_(source)


@dataclass
class _StepFieldStorage:
    host: torch.Tensor
    device: torch.Tensor


class CompiledStepFields:
    """Execution-owned generated kernel whose advance the executor repeats.

    The kernel IR is printed in the backend's dialect; ``requests`` describe
    its native compilation, which otherwise happens at the first launch.
    """

    def __init__(
        self,
        context: StepFieldCompileContext,
        *,
        backend: Backend,
        clock: torch.Tensor,
        duration: torch.Tensor,
        storage: Mapping[torch.dtype, torch.Tensor],
        executor: LoopExecutor,
    ) -> None:
        self.context = context
        self.clock = clock
        self.duration = duration
        self.tensors = (clock, duration, *(storage[dtype] for dtype in context.dtypes))
        self.executor = executor
        self.advance: Any = None
        self.module = None
        self.filename = None
        self.requests: tuple[CompileRequest, ...] = ()
        self.dialect = backend.dialect
        # MSL has no double: the clock's floating values take the widest
        # precision of the backend.
        real = torch.float64 if "float64" in backend.precisions else torch.float32
        function = step_field_program(context, real)
        try:
            {
                "torch": self._compile_torch,
                "triton": self._compile_triton,
                "cuda": self._compile_cuda,
                "msl": self._compile_metal,
            }[self.dialect](function)
        except BaseException:
            with cleanup_on_exit("step field compilation", (self.close,)):
                raise

    def _load_module(self, source: str):
        name = "hydroforge_step_time_" + uuid4().hex
        module = compile_generated_module(source, name=name)
        self.module = module
        self.filename = module.__file__
        self.source = source
        return module

    def _compile_torch(self, function: KernelFunction) -> None:
        # The source spells scalar parameters as literals: one function per
        # value of ``advance``.
        buffers = [param.name for param in function.params if param.access]
        states = dict(zip(buffers, self.tensors, strict=True))
        sizes = {name: tensor.numel() for name, tensor in states.items()}
        printer = TorchPrinter("device")
        names = {
            False: f"{function.name}_refresh",
            True: f"{function.name}_advance",
        }
        source = "\n\n\n".join(
            (
                TORCH_PRELUDE.rstrip("\n"),
                f"device = torch.device({str(self.clock.device)!r})",
                *(
                    printer.kernel(
                        replace(function, name=name),
                        sizes=sizes,
                        scalars={"advance": int(advance)},
                    )
                    for advance, name in names.items()
                ),
            )
        )
        module = self._load_module(source + "\n")
        kernels = {advance: getattr(module, name) for advance, name in names.items()}
        # The kernels read no sample phase.
        self.launch = lambda advance: kernels[advance](states, 0)

    def _compile_triton(self, function: KernelFunction) -> None:
        from hydroforge.kernels.toolchain.triton import warming_up, warmup_request
        from hydroforge.kernels.triton import launch_triton_kernel

        module = self._load_module(TRITON.program((function,)))
        launch = launch_triton_kernel(
            getattr(module, function.name), (1,), physics=False
        )
        self.launch = lambda advance: launch(
            *self.tensors, advance, BLOCK_SIZE=1, num_warps=1
        )

        def warmup() -> None:
            with warming_up():
                for advance in (False, True):
                    self.launch(advance)

        self.requests = (warmup_request(warmup, cost=len(self.source)),)

    def _compile_cuda(self, function: KernelFunction) -> None:
        from hydroforge.kernels.toolchain import cuda as rtc

        self.source = CUDA.program((function,))
        program = rtc.RtcProgram(
            self.source, rtc.program_options((), physics=False), "hydroforge_step_time"
        )
        device = self.clock.device.index
        request = rtc.RtcRequest(program, (function.name,))
        launches = {}

        def launch(advance: bool) -> None:
            prepared = launches.get(advance)
            if prepared is None:
                step = rtc.CudaLaunch(
                    function.name,
                    1,
                    1,
                    (*map(rtc.pointer, self.tensors), rtc.boolean(advance)),
                )
                prepared = launches[advance] = rtc.prepare(request, (step,), device)
            prepared()

        self.requests = (rtc.precompile_request(request, device),)
        self.launch = launch

    def _compile_metal(self, function: KernelFunction) -> None:
        from hydroforge.kernels.metal import MetalArgument, MetalProgram

        self.source = MSL.program((function,))
        fields = msl_arguments(function)
        program = MetalProgram(
            self.source,
            function.name,
            tuple(MetalArgument(*field) for field in fields),
            extent=(),
        )
        names = [field[0] for field in fields]
        launches = {
            advance: program.specialize(
                dict(zip(names, (*self.tensors, advance), strict=True)), 1
            )
            for advance in (False, True)
        }
        self.launch = lambda advance: launches[advance]()

    def run(self, *, advance: bool) -> None:
        if not advance:
            self.launch(False)
            return
        if self.advance is None:
            self.advance = self.executor.repeated(
                partial(self.launch, True), mutated_state=self.tensors
            )
        # Clock updates are non-differentiable. A capture and its replays also
        # touch generator state that earlier inference captures may own.
        with torch.inference_mode():
            self.advance.run()

    def close(self) -> None:
        try:
            if self.advance is not None:
                advance, self.advance = self.advance, None
                advance.close()
        finally:
            if self.filename is not None:
                release_generated_module(self.module.__name__, self.filename)
                self.filename = None
            self.module = None
            self.launch = None
            self.tensors = ()


class StepFieldRuntime:
    """Advance continuous time on device; upload only changed external controls.

    Builtins and expression dependencies are fused into one scalar aggregator.
    Host callbacks are an explicit escape hatch and never supply builtins.
    """

    def __init__(self, plan: Any, *, executor: LoopExecutor) -> None:
        self.device = plan.device
        self.dtype = plan.dtype
        self.backend = plan.backend
        self.executor = executor
        self.providers = plan.step_fields.providers
        self._calendar_available = (
            not hasattr(plan, "initial_time")
            or plan.initial_time is not None
            or plan.schedule is not None
        )
        self.expressions = plan.step_fields.expressions
        sources = (*_BUILTIN_STEP_FIELDS, *self.expressions, *self.providers)
        self.slots = {name: index for index, name in enumerate(sources)}
        self.storage: dict[torch.dtype, _StepFieldStorage] = {}
        self.buffers: dict[tuple[str, torch.dtype], torch.Tensor] = {}
        self.identities: set[int] = set()
        self.time: StepTime | None = None
        self.values: dict[str, int | float] = {}
        self.clock: torch.Tensor | None = None
        self.duration: torch.Tensor | None = None
        self.program: CompiledStepFields | None = None
        self._expected: Any = None
        self._duration_value: tuple[int, int] | None = None
        self._calendar_key: tuple[str, bool] | None = None
        self._calendar_active = False
        self._device_demand = False
        self._requested: CompiledStepFields | None = None
        self._deferred = False

    def concrete_dtype(self, field: StepField) -> torch.dtype:
        return self.dtype if field.dtype == "precision" else getattr(torch, field.dtype)

    def bind_many(self, fields: Iterable[StepField]) -> None:
        fields = tuple(
            field
            for field in fields
            if (field.source, self.concrete_dtype(field)) not in self.buffers
        )
        if not fields:
            return
        demanded = set()

        def visit(source):
            if source in demanded:
                return
            if source not in self.slots:
                raise ValueError(f"unknown step field source {source!r}")
            demanded.add(source)
            if source in self.expressions:
                for dependency in self.expressions[source].dependencies:
                    visit(dependency)

        for field in fields:
            visit(field.source)
            dtype = self.concrete_dtype(field)
            kind = (
                "index" if dtype == torch.int64 else str(dtype).removeprefix("torch.")
            )
            self.backend.validate_scalars("step fields", {field.source: kind})
        if not self._calendar_available and any(
            source in _BUILTIN_STEP_FIELDS and source != "step_seconds"
            for source in demanded
        ):
            raise ValueError(
                "calendar step fields require initial_time or a simulation_schedule"
            )
        with _disable_current_modes(), torch.inference_mode(False):
            changed = False
            for field in fields:
                changed = self._allocate(field) or changed
            if changed and self.time is not None:
                self._refresh()

    def bind(self, field: StepField) -> torch.Tensor:
        self.bind_many((field,))
        return self.buffers[(field.source, self.concrete_dtype(field))]

    def _allocate(self, field: StepField) -> bool:
        if field.source not in self.slots:
            raise ValueError(f"unknown step field source {field.source!r}")
        dtype = self.concrete_dtype(field)
        key = (field.source, dtype)
        if key in self.buffers:
            return False
        if dtype not in self.storage:
            host = torch.zeros(len(self.slots), dtype=dtype)
            device = torch.zeros_like(host, device=self.device)
            self.storage[dtype] = _StepFieldStorage(host, device)
        slot = self.slots[field.source]
        buffer = self.storage[dtype].device[slot : slot + 1]
        self.buffers[key] = buffer
        self.identities.add(id(buffer))
        self._device_demand = self._device_demand or field.source not in self.providers
        self.invalidate_program()
        return True

    def _load_values(self) -> None:
        custom = {
            source: self.providers[source](self.time)
            for source in dict.fromkeys(source for source, _dtype in self.buffers)
            if source in self.providers and source not in self.values
        }
        if custom:
            checked = _StepFieldValues(values=custom).values
            for source, dtype in self.buffers:
                if source not in checked:
                    continue
                value = checked[source]
                if dtype in {torch.int32, torch.int64}:
                    limits = torch.iinfo(dtype)
                    if type(value) is not int or not limits.min <= value <= limits.max:
                        raise ValueError(
                            f"provider {source!r} must return an integer within {dtype} range"
                        )
                else:
                    normalize_floating_scalar(f"provider {source!r}", value, dtype)
            self.values.update(checked)

    def _upload_custom(self) -> None:
        dtypes = set()
        for source, dtype in self.buffers:
            if source in self.providers:
                self.storage[dtype].host[self.slots[source]] = self.values[source]
                dtypes.add(dtype)
        first = len(self.slots) - len(self.providers)
        for dtype in dtypes:
            storage = self.storage[dtype]
            _upload(storage.device[first:], storage.host[first:])

    def _date_key(self) -> tuple[str, bool]:
        date = self.time.current_time
        calendar = getattr(date, "calendar", "proleptic_gregorian")
        calendar = {
            "gregorian": "standard",
            "365_day": "noleap",
            "366_day": "all_leap",
        }.get(calendar, calendar)
        return calendar, getattr(date, "has_year_zero", True)

    def _context(self) -> StepFieldCompileContext:
        calendar, has_year_zero = self._date_key()
        return StepFieldCompileContext(
            calendar=calendar,
            has_year_zero=has_year_zero,
            outputs=tuple(
                StepFieldOutput(source, dtype, self.slots[source])
                for source, dtype in self.buffers
                if source not in self.providers
            ),
            expressions=self.expressions,
        )

    def _anchor(self, context: StepFieldCompileContext) -> None:
        date = self.time.current_time
        if date is None:
            year, ordinal, micros = 1, 0, 0
        else:
            year = date.year + (not context.has_year_zero and date.year < 0)
            start = date.replace(
                month=1, day=1, hour=0, minute=0, second=0, microsecond=0
            )
            ordinal = (date - start).days
            micros = (
                (date.hour * 60 + date.minute) * 60 + date.second
            ) * 1_000_000 + date.microsecond
        values = torch.tensor(
            (year, ordinal, micros, *self._duration_value), dtype=torch.int64
        )
        if self.clock is None:
            self.clock = values.to(self.device)
        else:
            _upload(self.clock, values)

    def _update_duration(self) -> None:
        delta = timedelta(seconds=self.time.step_seconds)
        value = (delta.days, delta.seconds * 1_000_000 + delta.microseconds)
        if value == self._duration_value:
            return
        host = torch.tensor(value, dtype=torch.int64)
        if self.duration is None:
            self.duration = host.to(self.device)
        else:
            _upload(self.duration, host)
        self._duration_value = value

    def _ensure_program(self, *, anchor: bool = False) -> CompiledStepFields | None:
        if not self._device_demand:
            return None
        key = self._date_key()
        if self.program is not None and key == self._calendar_key:
            self._update_duration()
            if anchor and self._calendar_active:
                self._anchor(self.program.context)
            return self.program
        context = self._context()
        if not context.outputs:
            return None
        calendar_needed = context.requires_calendar
        if calendar_needed:
            self.time.date
        if key != self._calendar_key:
            self.invalidate_program()
            anchor = True
        self._update_duration()
        if self.clock is None or (
            calendar_needed and (anchor or not self._calendar_active)
        ):
            self._anchor(context)
        self._calendar_key = key
        self._calendar_active = calendar_needed
        if self.program is None:
            self.program = CompiledStepFields(
                context,
                backend=self.backend,
                clock=self.clock,
                duration=self.duration,
                storage={dtype: slab.device for dtype, slab in self.storage.items()},
                executor=self.executor,
            )
        return self.program

    def _refresh(self) -> None:
        self._load_values()
        program = self._ensure_program()
        self._upload_custom()
        if program is None:
            return
        # A recording launches nothing until its compile batch, which then
        # also compiles this program; :meth:`flush` evaluates it after that.
        if recording_sink() is not None:
            self._deferred = True
        else:
            program.run(advance=False)

    def take_requests(self) -> tuple[CompileRequest, ...]:
        """The native compilation of the current program, once."""

        program = self.program
        if program is None or program is self._requested:
            return ()
        self._requested = program
        return program.requests

    def flush(self) -> None:
        """Evaluate the fields whose refresh a recording deferred."""

        if not self._deferred:
            return
        self._deferred = False
        if self.program is not None:
            with _disable_current_modes(), torch.inference_mode(False):
                self.program.run(advance=False)

    def prepare(self, current_time: Any, step_seconds: float) -> None:
        continuous = (
            self.clock is not None
            and self._expected is not None
            and current_time == self._expected
        )
        self.time = StepTime(current_time, step_seconds)
        continuous = continuous and self._date_key() == self._calendar_key
        self.values.clear()
        if self.buffers:
            with _disable_current_modes(), torch.inference_mode(False):
                self._load_values()
                program = self._ensure_program(anchor=not continuous)
                self._upload_custom()
                if program is not None:
                    self._deferred = False
                    program.run(advance=continuous)
        self._expected = (
            None
            if current_time is None
            else current_time + timedelta(seconds=step_seconds)
        )

    def invalidate_program(self) -> None:
        self._requested = None
        if self.program is not None:
            program, self.program = self.program, None
            program.close()

    def close(self) -> None:
        try:
            self.invalidate_program()
        finally:
            self.buffers.clear()
            self.storage.clear()
            self.identities.clear()
            self.values.clear()
            self.time = None
            self.clock = None
            self.duration = None
            self._expected = None
            self._duration_value = None
            self._calendar_key = None
            self._calendar_active = False
            self._device_demand = False
            self._deferred = False

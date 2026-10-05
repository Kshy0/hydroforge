"""Metal indirect command buffers over online-lowered substep programs.

Recorded ATen operators lower strictly to online Metal kernels (never to an
eager fallback) and registered kernels record through their dispatchers, so
one loop iteration replays as one command buffer.
"""

from __future__ import annotations

from collections.abc import Callable
from contextvars import ContextVar
from functools import partial
from typing import Any

import torch

from hydroforge.core.errors import SubstepCompileError, cleanup_on_exit
from hydroforge.execution.aten import (
    COMPILED_ATEN_CONTRACTS,
    normalize_fill_scalar,
    normalize_float32_scalar,
)
from hydroforge.execution.executors.base import DirectRunner, LoopExecutor
from hydroforge.execution.executors.eager import host_adaptive
from hydroforge.execution.operators import (
    CollectiveOperator,
    CompiledKernelCall,
    PredicateLoopOperator,
)
from hydroforge.kernels.backends import metal_aten
from hydroforge.kernels.backends.metal_control import (
    adaptive_control_commands,
    fixed_control_command,
    statistics_control_command,
)
from hydroforge.kernels.metal import MetalCommand, MetalCommandNode
from hydroforge.kernels.toolchain.metal import (
    MetalCommandSequence,
    record_metal_commands,
)
from hydroforge.statistics.phases import (
    CONTROL_FLAGS,
    CONTROL_PHASE,
    CONTROL_WEIGHT,
    sample_phase_expr,
)

# ====================================================================== #
# ATen lowering
# ====================================================================== #

_scatter_error: ContextVar[list[torch.Tensor | None] | None] = ContextVar(
    "hydroforge_metal_scatter_error",
    default=None,
)


def _scatter_error_flag(device: torch.device) -> torch.Tensor:
    shared = _scatter_error.get()
    if shared is not None and shared[0] is not None:
        return shared[0]
    flag = torch.zeros(1, dtype=torch.int32, device=device)
    if shared is not None:
        shared[0] = flag
    return flag


def _copy(output: torch.Tensor, source: torch.Tensor) -> MetalCommand:
    return MetalCommand(
        metal_aten.copy_program(output.dtype),
        {"output_ptr": output, "input_ptr": source, "n": output.numel()},
    )


def _lower_copy(operator: Any) -> tuple[MetalCommand, ...]:
    args, _kwargs, _output = operator.static_values()
    return (_copy(args[0], args[1]),)


def _lower_lerp(operator: Any) -> tuple[MetalCommand, ...]:
    args, kwargs, output = operator.static_values()
    start, end, weight = args[:3]
    destination = kwargs.get("out", output)
    return (
        MetalCommand(
            metal_aten.lerp_program(),
            {
                "input_ptr": start,
                "end_ptr": end,
                "weight_ptr": weight,
                "output_ptr": destination,
                "n": destination.numel(),
            },
        ),
    )


def _lower_scatter(operator: Any) -> tuple[MetalCommand, ...]:
    args, kwargs, output = operator.static_values()
    name = operator.function._schema.name
    destination, _dim, index, source = args[:4]
    alpha = 1.0
    if name == "aten::index_add_":
        alpha = kwargs.get("alpha", args[4] if len(args) > 4 else 1)
        alpha = normalize_float32_scalar("index_add_ alpha", alpha)
    calls: list[MetalCommand] = []
    target = destination if name.endswith("_") else output
    if not isinstance(target, torch.Tensor):
        raise SubstepCompileError("Metal scatter_add output is not address-stable")
    if target is not destination:
        calls.append(_copy(target, destination))
    error = _scatter_error_flag(index.device)
    calls.append(
        MetalCommand(
            metal_aten.scatter_add_program(index.dtype),
            {
                "output_ptr": target,
                "index_ptr": index,
                "source_ptr": source,
                "n": source.numel(),
                "output_n": target.numel(),
                "error_ptr": error,
                "alpha": alpha,
            },
            (error,),
        )
    )
    return tuple(calls)


def _lower_zero(operator: Any) -> tuple[MetalCommand, ...]:
    args, _kwargs, _output = operator.static_values()
    destination = args[0]
    return (
        MetalCommand(
            metal_aten.zero_program(destination.dtype),
            {"output_ptr": destination, "n": destination.numel()},
        ),
    )


def _lower_fill(operator: Any) -> tuple[MetalCommand, ...]:
    args, _kwargs, _output = operator.static_values()
    destination, value = args[:2]
    value = normalize_fill_scalar(destination.dtype, value)
    return (
        MetalCommand(
            metal_aten.fill_program(destination.dtype),
            {"output_ptr": destination, "value": value, "n": destination.numel()},
        ),
    )


def _lower_binary(operator: Any) -> tuple[MetalCommand, ...]:
    args, kwargs, output = operator.static_values()
    schema_name = operator.function._schema.name
    name = schema_name.removeprefix("aten::").removesuffix("_")
    left, right = args[:2]
    alpha = kwargs.get("alpha", args[2] if len(args) > 2 else 1)
    scaled = name in {"add", "sub"}
    if scaled:
        alpha = normalize_float32_scalar(f"{name} alpha", alpha)
    destination = left if schema_name.endswith("_") else output
    if not isinstance(destination, torch.Tensor):
        raise SubstepCompileError(f"Metal {name} output is not address-stable")
    result_dtype = torch.bool if name == "lt" else torch.float32
    rhs_kind = "tensor" if isinstance(right, torch.Tensor) else "scalar"
    arguments: dict[str, Any] = {
        "input_ptr": left,
        "output_ptr": destination,
        "n": destination.numel(),
    }
    if isinstance(right, torch.Tensor):
        if right.numel() == 1:
            rhs_kind = "tensor_scalar"
        arguments["rhs_ptr"] = right
    else:
        arguments["rhs"] = normalize_float32_scalar(name, right)
    if scaled:
        arguments["alpha"] = alpha
    return (
        MetalCommand(
            metal_aten.binary_program(name, rhs_kind, result_dtype, scaled),
            arguments,
        ),
    )


_SEMANTIC_LOWERERS = {
    "binary": _lower_binary,
    "copy": _lower_copy,
    "fill": _lower_fill,
    "lerp": _lower_lerp,
    "scatter": _lower_scatter,
    "zero": _lower_zero,
}
_contract_semantics = {
    contract.semantics for contract in COMPILED_ATEN_CONTRACTS.values()
}
if _contract_semantics != set(_SEMANTIC_LOWERERS):
    missing = sorted(_contract_semantics.difference(_SEMANTIC_LOWERERS))
    extra = sorted(set(_SEMANTIC_LOWERERS).difference(_contract_semantics))
    raise RuntimeError(
        "compiled ATen semantics differ between graph and Metal backends: "
        f"missing={missing}, extra={extra}"
    )
_contract_binary_names = {
    name.removeprefix("aten::").removesuffix("_")
    for name, contract in COMPILED_ATEN_CONTRACTS.items()
    if contract.semantics == "binary"
}
if _contract_binary_names != set(metal_aten.BINARY_EXPRESSIONS):
    missing = sorted(_contract_binary_names.difference(metal_aten.BINARY_EXPRESSIONS))
    extra = sorted(
        set(metal_aten.BINARY_EXPRESSIONS).difference(_contract_binary_names)
    )
    raise RuntimeError(
        "compiled binary ATen contract differs from Metal expressions: "
        f"missing={missing}, extra={extra}"
    )
_ATEN_LOWERERS = {
    name: _SEMANTIC_LOWERERS[contract.semantics]
    for name, contract in COMPILED_ATEN_CONTRACTS.items()
}


def _metal_lowerer(operator: Any) -> Callable:
    """Resolve backend capability before any command or scratch allocation."""
    name = operator.function._schema.name
    contract = COMPILED_ATEN_CONTRACTS.get(name)
    if contract is None:
        raise SubstepCompileError(
            f"Torch operator {name!r} has no native Metal ICB lowering"
        )
    args, kwargs, output = operator.static_values()
    semantics = contract.semantics
    if semantics in {"binary", "lerp"}:
        if args[0].dtype != torch.float32:
            raise SubstepCompileError(
                f"Metal {name} lowering requires float32 model precision"
            )
    elif args[0].dtype == torch.float64:
        raise SubstepCompileError(
            f"Metal {name} lowering requires a Metal-supported tensor dtype"
        )
    return _ATEN_LOWERERS[name]


def lower_metal_aten(operator: Any) -> tuple[MetalCommand, ...]:
    """Lower one recorded ATen node; never return an eager fallback."""
    return _metal_lowerer(operator)(operator)


def _lower_program(program: Any) -> tuple[Any, ...]:
    commands: list[Any] = []
    for operator in program.operators:
        if isinstance(operator, CompiledKernelCall):
            commands.append(operator)
        elif isinstance(operator, CollectiveOperator):
            raise SubstepCompileError(
                "Metal ICB substeps do not support distributed collectives"
            )
        elif isinstance(operator, PredicateLoopOperator):
            raise SubstepCompileError(
                "Metal ICB substeps do not support predicate loops"
            )
        else:
            commands.extend(lower_metal_aten(operator))
    return tuple(commands)


class LoweredPrograms:
    """Metal commands of programs launched together.

    Their scatter lowerings share one bounds-error flag, which the lowering
    owns: it must outlive every command buffer recorded from the commands.
    """

    def __init__(self, *programs: Any) -> None:
        for program in programs:
            for operator in program.operators:
                if isinstance(operator, (CollectiveOperator, PredicateLoopOperator)):
                    raise SubstepCompileError(
                        "Metal ICB substeps do not support collectives or predicate loops"
                    )
                if not isinstance(operator, CompiledKernelCall):
                    _metal_lowerer(operator)
        token = _scatter_error.set([None])
        try:
            self.commands = tuple(_lower_program(program) for program in programs)
        finally:
            _scatter_error.reset(token)
        self.errors = tuple(
            dict.fromkeys(
                flag
                for commands in self.commands
                for command in commands
                for flag in getattr(command, "errors", ())
            )
        )

    def reset(self) -> None:
        for flag in self.errors:
            flag.zero_()

    def check(self) -> None:
        """Read every distinct scatter bounds flag once for one launch."""

        if any(int(flag.item()) != 0 for flag in self.errors):
            raise IndexError("Metal scatter_add index is outside the output extent")


class _StatisticsOperator(MetalCommandNode):
    """Record one generated Metal aggregator at its substep sequence point."""

    def __init__(self, launch: Any) -> None:
        self.launch = launch
        tensors = tuple(dict.fromkeys(launch.states.values()))
        # Exact hazards are derived again by each Metal dispatcher while the
        # aggregator wrapper records. This conservative boundary covers the
        # wrapper as one operator relative to adjacent physics/control nodes.
        self.reads = tensors
        self.writes = tensors

    def record(self) -> None:
        self.launch(-1)


# ====================================================================== #
# Executor and runners
# ====================================================================== #


class MetalIcbExecutor(LoopExecutor):
    """Replay loop iterations and outer programs as Metal command buffers."""

    name = "metal_icb"

    def capture_commands(self, commands: tuple[Any, ...], *, cyclic: bool = False):
        """Compile one ordered command tuple into a single owned command buffer.

        ``cyclic`` places a barrier after the final command as well, making the
        command buffer safe to encode repeatedly in one command encoder.
        """

        sequence = MetalCommandSequence()
        with record_metal_commands(sequence):
            for command in commands:
                command.record()
            if cyclic:
                sequence.mark_barrier()
        return self.register(sequence.capture())

    def fixed(self, loop: Any) -> _Fixed:
        return _Fixed(self, loop)

    def adaptive(self, loop: Any) -> _Adaptive:
        return _Adaptive(self, loop)

    def predicate(self, loop: Any) -> Any:
        raise SubstepCompileError("Metal ICB substeps do not support predicate loops")

    def outer(self, program: Any) -> _Outer:
        return _Outer(self, program)

    def repeated(
        self, body: Callable[[], None], *, mutated_state: tuple[Any, ...]
    ) -> DirectRunner:
        del mutated_state
        return DirectRunner(body)

    def sample(self, launch: Any, phase: int) -> None:
        launch(phase)


class _Fixed:
    """Replay a fixed loop's iteration command buffer ``count`` times."""

    def __init__(self, executor: MetalIcbExecutor, loop: Any) -> None:
        self.executor = executor
        self.loop = loop
        self.lowered = LoweredPrograms(*loop.programs)
        self.iterations = self._capture(
            (self._control(),), scope="fixed Metal final capture"
        )
        # (statistics launch, iteration pair folding it)
        self.folded: tuple[Any, tuple[Any, Any | None]] | None = None
        self.width = None
        self._width: float | None = None

    def _control(self) -> MetalCommand:
        loop = self.loop
        return fixed_control_command(
            count=loop.count, counter=loop.counter, continue_flag=loop.continue_flag
        )

    def _capture(self, tail: tuple[Any, ...], *, scope: str) -> tuple[Any, Any | None]:
        """Capture regular/final iterations; roll back a partial pair."""

        executor = self.executor
        body = self.lowered.commands[0]
        regular = executor.capture_commands((*body, *tail), cyclic=True)
        if self.loop.final is None:
            return regular, None
        try:
            final = executor.capture_commands(
                (*body, *self.lowered.commands[1], *tail), cyclic=True
            )
        except BaseException:
            with cleanup_on_exit(scope, (partial(executor.release, regular),)):
                raise
        return regular, final

    def _folded(self, launch: Any) -> tuple[Any, Any | None]:
        if self.folded is not None and self.folded[0] is launch:
            return self.folded[1]
        previous, self.folded = self.folded, None
        if previous is not None:
            self.executor.release_all(
                previous[1], scope="fixed Metal statistics replacement"
            )
        loop = self.loop
        states = launch.states
        control = statistics_control_command(
            sample_phase=sample_phase_expr,
            weight_source=self.width,
            continue_flag=loop.continue_flag,
            counter=loop.counter,
            flags=states[CONTROL_FLAGS],
            weight=states[CONTROL_WEIGHT],
            phase=states[CONTROL_PHASE],
        )
        pair = self._capture(
            (self._control(), control, _StatisticsOperator(launch)),
            scope="fixed Metal statistics final capture",
        )
        self.folded = (launch, pair)
        return pair

    def run(self, count: int, duration: float, step: Any) -> int:
        loop = self.loop
        self.lowered.reset()
        fold = step.fold
        loop.prepare(
            count, duration / count, controls=True, weight=fold or loop.weighted
        )
        if fold and self.lowered.errors:
            # Bounds flags are host-visible only after replay. Keep the same
            # per-iteration sampling semantics, checking before each sample.
            regular, final = self.iterations
            for index in range(count):
                (
                    final if final is not None and index == count - 1 else regular
                ).replay()
                self.lowered.check()
                step.sample(
                    first=index == 0, last=index == count - 1, weight=duration / count
                )
            return count
        if fold:
            step.statistics.prelaunch()
            launch = step.statistics.launch
            dtype = launch.states[CONTROL_WEIGHT].dtype
            if self.width is None or self.width.dtype != dtype:
                if dtype == torch.float64:
                    from hydroforge.kernels.emulated import EmulatedTensor

                    self.width = EmulatedTensor.encode(
                        torch.zeros(1, dtype=torch.float64, device="cpu"),
                        self.executor.device,
                    )
                else:
                    self.width = torch.zeros(
                        1, dtype=dtype, device=self.executor.device
                    )
                self._width = None
            width = duration / count
            if width != self._width:
                self.width.fill_(width)
                self._width = width
            regular, final = self._folded(launch)
        else:
            regular, final = self.iterations
        if final is None:
            regular.replay(count)
        else:
            if count > 1:
                regular.replay(count - 1)
            final.replay()
        self.lowered.check()
        if step.sampling and not fold:
            step.sample(first=count == 1, last=True, weight=duration)
        return count

    def close(self) -> None:
        folded, self.folded = self.folded, None
        iterations, self.iterations = self.iterations, ()
        # Command buffers reference the lowering scratch: release them first.
        try:
            self.executor.release_all(
                (*iterations, *(() if folded is None else folded[1])),
                scope="fixed Metal command buffers",
            )
        finally:
            self.lowered = None


class _Adaptive:
    """Replay an adaptive iteration's command buffer under host control."""

    def __init__(self, executor: MetalIcbExecutor, loop: Any) -> None:
        self.executor = executor
        self.loop = loop
        self.lowered = LoweredPrograms(loop.proposal, loop.body)
        begin, accept, end = adaptive_control_commands(
            candidate=loop.candidate,
            maximum=loop.maximum,
            duration=loop.duration,
            elapsed=loop.elapsed,
            dt=loop.time_step,
            counter=loop.counter,
            continue_flag=loop.continue_flag,
            error_flag=loop.error_flag,
            status=loop.status,
            maximum_steps=loop.maximum_steps,
        )
        proposal, body = self.lowered.commands
        self.iteration = executor.capture_commands(
            (begin, *proposal, accept, *body, end), cyclic=True
        )

    def run(self, duration: float, step: Any) -> int:
        self.lowered.reset()

        # The end command writes ``status`` inside the command buffer.
        def iterate():
            self.iteration.replay()
            self.lowered.check()

        count = host_adaptive(self.loop, step, iterate, ())
        self.lowered.check()
        return count

    def close(self) -> None:
        iteration, self.iteration = self.iteration, None
        try:
            if iteration is not None:
                self.executor.release(iteration)
        finally:
            self.lowered = None


class _Outer:
    """Replay an outer program as one command buffer."""

    def __init__(self, executor: MetalIcbExecutor, program: Any) -> None:
        self.executor = executor
        self.lowered = LoweredPrograms(program)
        (commands,) = self.lowered.commands
        self.icb = executor.capture_commands(commands) if commands else None

    def run(self) -> None:
        self.lowered.reset()
        if self.icb is not None:
            self.icb.replay()
        self.lowered.check()

    def close(self) -> None:
        icb, self.icb = self.icb, None
        try:
            if icb is not None:
                self.executor.release(icb)
        finally:
            self.lowered = None

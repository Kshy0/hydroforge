"""Execution-owned statistics observation and device-fold coordination."""

from __future__ import annotations

from typing import Any

import torch

from hydroforge.kernels.backends.metal.protocol import MetalCommandNode


class _MetalStatisticsOperator(MetalCommandNode):
    """Record one generated Metal aggregator at its substep sequence point."""

    def __init__(self, observer: StatisticsObserver) -> None:
        self.observer = observer
        states = observer.aggregator._kernel_states
        tensors = tuple(
            dict.fromkeys(
                value for value in states.values()
                if isinstance(value, torch.Tensor)
            )
        )
        # Exact hazards are derived again by each Metal dispatcher while the
        # aggregator wrapper records. This conservative boundary covers the
        # wrapper as one operator relative to adjacent physics/control nodes.
        self.reads = tensors
        self.writes = tensors

    def record(self) -> None:
        aggregator = self.observer.aggregator
        aggregator._aggregator_function(
            aggregator._kernel_states, self.observer.model.BLOCK_SIZE,
        )


class StatisticsObserver:
    """Attach statistics to a model execution without model-side bookkeeping."""

    _FOLD_INNER_OPS = frozenset({"last", "mean", "sum", "max", "min", "first"})

    def __init__(self, model: Any) -> None:
        self.model = model
        self.aggregator: Any = None
        self._fold_policy_cache: tuple[bool, bool] | None = None

    def attach(self, aggregator: Any) -> None:
        self.aggregator = aggregator
        self._fold_policy_cache = None

    def detach(self, aggregator: Any) -> None:
        """Release the current aggregator without accepting stale owners."""
        if self.aggregator is not aggregator:
            raise RuntimeError("cannot detach an aggregator not owned by this execution")
        self.aggregator = None
        self._fold_policy_cache = None

    def enabled(self, output_enabled: bool) -> bool:
        return bool(output_enabled and self.aggregator is not None)

    def sample(
        self,
        *,
        sub_step: int,
        num_sub_steps: int,
        flags: int,
        weight: float,
        total_weight: float,
    ) -> None:
        aggregator = self.aggregator
        if aggregator is not None:
            aggregator.update_statistics(
                sub_step, num_sub_steps, flags, weight, total_weight,
            )

    def finish(self, current_time: Any) -> None:
        aggregator = self.aggregator
        if aggregator is not None:
            aggregator.finalize_time_step(current_time)

    def check_background_failures(self, current_time: Any) -> None:
        aggregator = self.aggregator
        if aggregator is not None:
            aggregator.check_background_failures(current_time)

    def ensure_output_durable(self, current_time: Any) -> None:
        aggregator = self.aggregator
        if aggregator is not None:
            aggregator.ensure_output_durable(current_time)

    def _fold_policy(self) -> tuple[bool, bool]:
        cached = self._fold_policy_cache
        if cached is not None:
            return cached
        aggregator = self.aggregator
        if aggregator is None or not aggregator._aggregator_generated:
            return True, False
        reductions = tuple(
            (operation.inner or operation.outer).value
            for variable in aggregator._statistics_ir.variables
            for operation in variable.operations
        )
        compatible = all(
            reduction in self._FOLD_INNER_OPS for reduction in reductions
        )
        should_fold = any(
            reduction in self._FOLD_INNER_OPS and reduction != "last"
            for reduction in reductions
        )
        self._fold_policy_cache = (compatible, should_fold)
        return self._fold_policy_cache

    def device_compatible(self) -> bool:
        return self._fold_policy()[0]

    def should_fold(self) -> bool:
        return self._fold_policy()[1]

    def accumulators(self) -> list[torch.Tensor]:
        return [
            value
            for name, value in self.aggregator._kernel_states.items()
            if isinstance(value, torch.Tensor) and not name.startswith("__")
        ]

    def captured_body(
        self,
        *,
        graph: Any,
        weight_src: torch.Tensor,
        counter: torch.Tensor,
        continue_flag: torch.Tensor,
        stream_ptr: int,
    ) -> None:
        aggregator = self.aggregator
        states = aggregator._kernel_states
        graph.stats_control(
            weight_src=weight_src,
            continue_flag=continue_flag,
            counter=counter,
            weight=states["__weight"],
            sub_step=states["__sub_step"],
            num_sub_steps=states["__num_sub_steps"],
            stream_ptr=stream_ptr,
        )
        aggregator._aggregator_function(states, self.model.BLOCK_SIZE)

    def prelaunch(self, flags: int, total_weight: float) -> None:
        aggregator = self.aggregator
        states = aggregator._kernel_states
        is_inner_last = bool(flags & 2)
        is_outer_first = bool(flags & 4) and is_inner_last
        is_outer_last = bool(flags & 8) and is_inner_last
        if is_outer_first:
            aggregator._macro_step_index = 0
            aggregator._current_macro_step_count = 0.0
            aggregator._outer_flags_ever_seen = True
        if is_inner_last or is_outer_last:
            for name, outer in aggregator._output_is_outer.items():
                if (not outer and is_inner_last) or (outer and is_outer_last):
                    aggregator._dirty_outputs.add(name)
        if is_inner_last:
            aggregator._current_macro_step_count += 1.0
        states["__total_weight"].fill_(total_weight)
        states["__flags"].fill_(flags)
        states["__num_macro_steps"].fill_(aggregator._current_macro_step_count)
        states["__macro_step_index"].fill_(aggregator._macro_step_index)

    def metal_operator(self) -> _MetalStatisticsOperator:
        aggregator = self.aggregator
        if aggregator is None or not aggregator._aggregator_generated:
            raise RuntimeError(
                "Metal statistics folding requires a generated aggregator"
            )
        return _MetalStatisticsOperator(self)

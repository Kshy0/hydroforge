"""Execution-owned statistics observation and device-fold coordination."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from hydroforge.contracts.naming import RESERVED_CONTROL_STATE
from hydroforge.contracts.temporal import DateLike
from hydroforge.kernels.backends.metal.protocol import MetalCommandNode

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel
    from hydroforge.statistics.runtime import StatisticsRuntime


class _MetalStatisticsOperator(MetalCommandNode):
    """Record one generated Metal aggregator at its substep sequence point."""

    def __init__(self, observer: StatisticsObserver) -> None:
        self.observer = observer
        states = observer.aggregator._kernel_states
        tensors = tuple(dict.fromkeys(states.values()))
        # Exact hazards are derived again by each Metal dispatcher while the
        # aggregator wrapper records. This conservative boundary covers the
        # wrapper as one operator relative to adjacent physics/control nodes.
        self.reads = tensors
        self.writes = tensors

    def record(self) -> None:
        aggregator = self.observer.aggregator
        aggregator._aggregator_function(
            aggregator._kernel_states, aggregator.block_size,
        )


class StatisticsObserver:
    """Attach statistics to a model execution without model-side bookkeeping."""

    _FOLD_INNER_OPS = frozenset({"last", "mean", "sum", "max", "min", "first"})

    def __init__(
        self, model: AbstractModel, aggregator: StatisticsRuntime,
    ) -> None:
        self.model = model
        self.aggregator = aggregator
        self._fold_policy_cache: tuple[bool, bool] | None = None

    def invalidate(self) -> None:
        """Forget policies compiled from one live aggregator program."""
        self._fold_policy_cache = None

    def enabled(self, output_enabled: bool) -> bool:
        return output_enabled

    def sample(
        self,
        *,
        sub_step: int,
        num_sub_steps: int,
        flags: int,
        weight: float,
        total_weight: float,
    ) -> None:
        self.aggregator.update_statistics(
            sub_step, num_sub_steps, flags, weight, total_weight,
        )

    def finish(self, current_time: DateLike) -> None:
        self.aggregator.finalize_time_step(current_time)

    def check_background_failures(
        self, current_time: DateLike,
    ) -> None:
        self.aggregator.check_background_failures(current_time)

    def _fold_policy(self) -> tuple[bool, bool]:
        cached = self._fold_policy_cache
        if cached is not None:
            return cached
        aggregator = self.aggregator
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
            if name not in RESERVED_CONTROL_STATE
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
        aggregator._aggregator_function(states, aggregator.block_size)

    def prelaunch(self, flags: int, total_weight: float) -> None:
        aggregator = self.aggregator
        converted_total = aggregator._convert_control_float(
            "total_weight", total_weight,
        )
        states = aggregator._kernel_states
        is_inner_last = bool(flags & 2)
        is_outer_first = bool(flags & 4) and is_inner_last
        is_outer_last = bool(flags & 8) and is_inner_last
        num_macro_steps, macro_step_index = aggregator._claim_macro_step(
            is_inner_last=is_inner_last,
            is_outer_first=is_outer_first,
            is_outer_last=is_outer_last,
        )
        states["__total_weight"].fill_(converted_total)
        states["__flags"].fill_(flags)
        states["__num_macro_steps"].fill_(num_macro_steps)
        states["__macro_step_index"].fill_(macro_step_index)

    def metal_operator(self) -> _MetalStatisticsOperator:
        return _MetalStatisticsOperator(self)


class DisabledStatisticsObserver:
    """Fixed no-statistics execution policy for models without a declaration."""

    def __init__(self, model: AbstractModel) -> None:
        self.model = model

    def enabled(self, output_enabled: bool) -> bool:
        del output_enabled
        return False

    def sample(self, **values: Any) -> None:
        del values

    def finish(self, current_time: DateLike) -> None:
        del current_time

    def check_background_failures(self, current_time: DateLike) -> None:
        del current_time

    def device_compatible(self) -> bool:
        return True

    def should_fold(self) -> bool:
        return False

    def accumulators(self) -> tuple[()]:
        return ()

    def invalidate(self) -> None:
        pass

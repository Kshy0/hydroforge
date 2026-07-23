"""Conservative forcing resampling onto a driver-owned model schedule."""

from __future__ import annotations

from contextlib import contextmanager
import copy
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Callable, Mapping

import torch

from hydroforge.contracts.temporal import (
    DatasetTemporalContract,
    ForcingSemantics,
    ResamplingMethod,
    SimulationSchedule,
    SimulationStep,
    date_signature,
    timedelta_microseconds,
)


@dataclass(frozen=True, slots=True)
class ForcingSource:
    contract: DatasetTemporalContract
    semantics: ForcingSemantics
    resampling: ResamplingMethod

    def __post_init__(self) -> None:
        if not isinstance(self.contract, DatasetTemporalContract):
            raise TypeError(
                "forcing contract must be a DatasetTemporalContract"
            )
        if type(self.semantics) is not str:
            raise TypeError("forcing semantics must be a string")
        if type(self.resampling) is not str:
            raise TypeError("forcing resampling method must be a string")
        allowed = {
            "mean_rate": {"hold", "conservative"},
            "accumulated": {"conservative"},
            "instantaneous": {"linear"},
        }
        if self.semantics not in allowed:
            raise ValueError(f"unknown forcing semantics {self.semantics!r}")
        if self.resampling not in allowed[self.semantics]:
            raise ValueError(
                f"{self.semantics!r} forcing does not support "
                f"{self.resampling!r} resampling"
            )


@dataclass(frozen=True, slots=True)
class ForcingContribution:
    source_index: int
    weight: float


class ForcingPlan:
    """Prevalidated temporal mapping from source samples to model steps."""

    __slots__ = (
        "_schedule", "_sources", "_fingerprint", "_explicit_contributions",
    )

    def __init__(
        self,
        schedule: SimulationSchedule,
        sources: Mapping[str, ForcingSource],
    ) -> None:
        if not isinstance(sources, Mapping):
            raise TypeError("forcing sources must be a mapping")
        if not sources:
            raise ValueError("forcing plan requires at least one source")
        invalid_names = [
            name for name in sources
            if not isinstance(name, str) or not name
        ]
        if invalid_names:
            raise ValueError(
                f"forcing source names must be non-empty strings: {invalid_names!r}"
            )
        invalid_sources = {
            name: type(source).__name__
            for name, source in sources.items()
            if not isinstance(source, ForcingSource)
        }
        if invalid_sources:
            raise TypeError(
                f"forcing sources must be ForcingSource values: {invalid_sources}"
            )
        if not isinstance(schedule, SimulationSchedule):
            raise TypeError("forcing schedule must be a SimulationSchedule")
        self._schedule = schedule
        self._sources = MappingProxyType(dict(sources))
        self._fingerprint = self._compile_fingerprint()
        self._explicit_contributions: dict[
            str, tuple[tuple[ForcingContribution, ...], ...] | None
        ] = {
            name: self._compile_source(name, source)
            for name, source in self.sources.items()
        }

    @property
    def schedule(self) -> SimulationSchedule:
        return self._schedule

    @property
    def sources(self) -> Mapping[str, ForcingSource]:
        return self._sources

    @property
    def fingerprint(self) -> str:
        return self._fingerprint

    def _compile_fingerprint(self) -> str:
        definition = {
            "schedule": self.schedule.fingerprint,
            "sources": {
                name: {
                    "calendar": source.contract.calendar,
                    "start": date_signature(source.contract.start),
                    "interval_microseconds": timedelta_microseconds(
                        source.contract.interval,
                        label=f"forcing {name!r} interval",
                    ),
                    "count": source.contract.count,
                    "timestamp_position": source.contract.timestamp_position,
                    "semantics": source.semantics,
                    "resampling": source.resampling,
                }
                for name, source in sorted(self.sources.items())
            },
        }
        encoded = json.dumps(
            definition, sort_keys=True, separators=(",", ":"),
        )
        return hashlib.sha256(encoded.encode()).hexdigest()

    @classmethod
    def bind(
        cls,
        *,
        schedule: SimulationSchedule,
        sources: Mapping[str, ForcingSource] | None = None,
        **named_sources: ForcingSource,
    ) -> ForcingPlan:
        if sources is not None and not isinstance(sources, Mapping):
            raise TypeError("forcing sources must be a mapping")
        combined = dict(sources) if sources is not None else {}
        overlap = set(combined).intersection(named_sources)
        if overlap:
            raise ValueError(f"duplicate forcing sources: {sorted(overlap)}")
        combined.update(named_sources)
        return cls(schedule, combined)

    def _compile_source(
        self, name: str, source: ForcingSource,
    ) -> tuple[tuple[ForcingContribution, ...], ...] | None:
        contract = source.contract
        if contract.calendar != self.schedule.calendar:
            raise ValueError(
                f"forcing {name!r} calendar {contract.calendar!r} differs "
                f"from schedule calendar {self.schedule.calendar!r}"
            )
        if type(contract.start) is not type(self.schedule.start):
            raise TypeError(
                f"forcing {name!r} and simulation schedule must use the same "
                "datetime representation"
            )
        if self.schedule.is_regular:
            self._validate_regular_source(name, source)
            return None
        return tuple(
            self._linear(source, step)
            if source.resampling == "linear"
            else self._overlap(source, step)
            for step in self.schedule
        )

    def _validate_regular_source(
        self, name: str, source: ForcingSource,
    ) -> None:
        """Prove a regular source mapping without expanding model steps."""

        last_index = len(self.schedule) - 1
        edge_indices = (0,) if last_index == 0 else (0, last_index)
        for index in edge_indices:
            step = self.schedule.step_at(index)
            if source.resampling == "linear":
                self._linear(source, step)
            else:
                self._overlap(source, step)
        if source.resampling != "hold":
            return

        contract = source.contract
        start_offset = timedelta_microseconds(
            self.schedule.start - contract.start,
            label=f"forcing {name!r} start offset",
        )
        end_offset = timedelta_microseconds(
            self.schedule.end - contract.start,
            label=f"forcing {name!r} end offset",
        )
        source_width = timedelta_microseconds(
            contract.interval, label=f"forcing {name!r} interval",
        )
        cadence = timedelta_microseconds(
            self.schedule.cadence, label="simulation step",
        )
        first_boundary = (start_offset // source_width + 1) * source_width
        if first_boundary >= end_offset:
            return
        aligned = (first_boundary - start_offset) % cadence == 0
        multiple_boundaries = first_boundary + source_width < end_offset
        repeating_alignment = not multiple_boundaries or source_width % cadence == 0
        if not aligned or not repeating_alignment:
            raise ValueError(
                f"hold forcing {name!r} has a source boundary inside a model "
                "step; align the regular model cadence or use conservative "
                "resampling"
            )

    @staticmethod
    def _source_coordinate(
        contract: DatasetTemporalContract, time: Any,
    ) -> tuple[int, int, int]:
        """Return exact floor index, remainder, and interval in microseconds."""

        width = timedelta_microseconds(
            contract.interval, label="forcing source interval",
        )
        index, remainder = divmod(
            timedelta_microseconds(
                time - contract.start, label="forcing source offset",
            ),
            width,
        )
        return index, remainder, width

    def _linear(
        self, source: ForcingSource, step: SimulationStep,
    ) -> tuple[ForcingContribution, ...]:
        contract = source.contract
        midpoint = step.start + (step.end - step.start) / 2
        left, remainder, width = self._source_coordinate(contract, midpoint)
        fraction = remainder / width
        if left < 0 or left >= contract.count:
            raise ValueError(
                f"model step {step.index} is outside instantaneous forcing coverage"
            )
        if fraction == 0.0 or left == contract.count - 1:
            if fraction != 0.0 and left == contract.count - 1:
                raise ValueError(
                    f"model step {step.index} requires an instantaneous sample "
                    "after the source end"
                )
            return (ForcingContribution(left, 1.0),)
        return (
            ForcingContribution(left, 1.0 - fraction),
            ForcingContribution(left + 1, fraction),
        )

    def _overlap(
        self, source: ForcingSource, step: SimulationStep,
    ) -> tuple[ForcingContribution, ...]:
        contract = source.contract
        source_width = timedelta_microseconds(
            contract.interval, label="forcing source interval",
        )
        step_width = timedelta_microseconds(
            step.end - step.start, label="model step duration",
        )
        first, _remainder, width = self._source_coordinate(
            contract, step.start,
        )
        end_offset = timedelta_microseconds(
            step.end - contract.start, label="forcing end offset",
        )
        last = (end_offset + width - 1) // width - 1
        contributions: list[ForcingContribution] = []
        for index in range(first, last + 1):
            if not 0 <= index < contract.count:
                raise ValueError(
                    f"model step {step.index} is outside forcing coverage"
                )
            source_start, source_end = contract.support(index)
            overlap_start = max(step.start, source_start)
            overlap_end = min(step.end, source_end)
            overlap = max(0, timedelta_microseconds(
                overlap_end - overlap_start, label="forcing overlap",
            ))
            if overlap == 0:
                continue
            if source.resampling == "hold" and overlap < step_width:
                raise ValueError(
                    f"hold forcing {step.index} crosses a source boundary; "
                    "use conservative resampling"
                )
            weight = (
                overlap / step_width
                if source.semantics == "mean_rate"
                else overlap / source_width / (step_width / 1_000_000)
            )
            contributions.append(ForcingContribution(index, weight))
        if not contributions:
            raise ValueError(f"model step {step.index} has no forcing coverage")
        if source.semantics == "mean_rate" and not math.isclose(
            sum(item.weight for item in contributions), 1.0,
            rel_tol=0.0, abs_tol=1e-12,
        ):
            raise ValueError(f"forcing does not fully cover model step {step.index}")
        return tuple(contributions)

    def contributions(
        self, name: str, step: int | SimulationStep,
    ) -> tuple[ForcingContribution, ...]:
        if name not in self.sources:
            raise KeyError(name)
        index = self._step_index(step)
        compiled = self._explicit_contributions[name]
        if compiled is not None:
            return compiled[index]
        model_step = self.schedule.step_at(index)
        source = self.sources[name]
        return (
            self._linear(source, model_step)
            if source.resampling == "linear"
            else self._overlap(source, model_step)
        )

    def _step_index(self, step: int | SimulationStep) -> int:
        if type(step) is int:
            index = step
        elif type(step) is SimulationStep:
            index = step.index
            if not 0 <= index < len(self.schedule):
                raise IndexError(index)
            expected = self.schedule.step_at(index)
            if step != expected:
                raise ValueError(
                    f"model step {index} does not belong to this forcing "
                    f"schedule: expected [{expected.start}, {expected.end}), "
                    f"got [{step.start}, {step.end})"
                )
        else:
            raise TypeError("forcing step must be an exact int or SimulationStep")
        if not 0 <= index < len(self.schedule):
            raise IndexError(index)
        return index

    def resample(
        self,
        name: str,
        step: int | SimulationStep,
        read: Callable[[int], Any],
    ) -> Any:
        """Return a model-step mean rate from source samples.

        Accumulated source values are conservatively converted to a rate;
        mean-rate and instantaneous sources preserve their native units.
        """
        if not callable(read):
            raise TypeError("forcing reader must be callable")
        result = None
        contributions = self.contributions(name, step)
        for contribution in contributions:
            value = read(contribution.source_index)
            term = (
                value
                if len(contributions) == 1 and contribution.weight == 1.0
                else value * contribution.weight
            )
            result = term if result is None else result + term
        return result

    def stream(
        self,
        name: str,
        read: Callable[[int], Any],
        *,
        start_step: int = 0,
    ) -> ForcingStream:
        return ForcingStream(self, name, read, start_step=start_step)

    def bundle(
        self,
        readers: Mapping[str, Callable[[int], Any]] | None = None,
        *,
        start_step: int = 0,
        **named_readers: Callable[[int], Any],
    ) -> ForcingBundle:
        """Bind every declared source to one atomically advanced cursor."""

        if readers is not None and not isinstance(readers, Mapping):
            raise TypeError("forcing readers must be a mapping")
        combined = dict(readers) if readers is not None else {}
        overlap = set(combined).intersection(named_readers)
        if overlap:
            raise ValueError(f"duplicate forcing readers: {sorted(overlap)}")
        combined.update(named_readers)
        return ForcingBundle(self, combined, start_step=start_step)


class ForcingStream:
    """Sequential cached reader; source samples are loaded only when needed."""

    def __init__(
        self,
        plan: ForcingPlan,
        name: str,
        read: Callable[[int], Any],
        *,
        start_step: int = 0,
    ) -> None:
        if not isinstance(plan, ForcingPlan):
            raise TypeError("forcing stream plan must be a ForcingPlan")
        if name not in plan.sources:
            raise KeyError(name)
        if not callable(read):
            raise TypeError("forcing stream reader must be callable")
        self.plan = plan
        self.name = name
        self.read = read
        self._cache: dict[int, Any] = {}
        if type(start_step) is not int:
            raise TypeError("forcing stream start_step must be an exact int")
        if not 0 <= start_step <= len(plan.schedule):
            raise ValueError("forcing stream start_step is outside the schedule")
        self._next_step = start_step

    def checkpoint_state(self) -> dict[str, Any]:
        """Return a JSON-safe cursor bound to the exact temporal plan."""

        return {
            "fingerprint": self.plan.fingerprint,
            "source": self.name,
            "next_step": self._next_step,
        }

    def restore_checkpoint_state(self, state: Any) -> None:
        """Validate then restore a forcing cursor without retaining stale data."""

        if not isinstance(state, dict) or set(state) != {
            "fingerprint", "source", "next_step",
        }:
            raise ValueError("forcing stream checkpoint has an invalid schema")
        if state["fingerprint"] != self.plan.fingerprint:
            raise ValueError("forcing stream checkpoint plan does not match")
        if state["source"] != self.name:
            raise ValueError("forcing stream checkpoint source does not match")
        next_step = state["next_step"]
        if type(next_step) is not int or not 0 <= next_step <= len(
            self.plan.schedule,
        ):
            raise ValueError("forcing stream checkpoint cursor is outside schedule")
        self._next_step = next_step
        self._cache.clear()

    def read_step(self, step: int | SimulationStep) -> Any:
        index = self.plan._step_index(step)
        if index != self._next_step:
            raise ValueError(
                f"forcing stream expected model step {self._next_step}, got {index}"
            )
        result, retained = self._stage(index)
        self._commit(index, retained)
        return result

    def _stage(
        self, index: int, *, isolate_consumer: bool = False,
    ) -> tuple[Any, dict[int, Any]]:
        """Read one value and precompute a no-fail cursor commit."""

        contributions = self.plan.contributions(self.name, index)
        result = None
        for contribution in contributions:
            if contribution.source_index not in self._cache:
                value = self.read(contribution.source_index)
                self._cache[contribution.source_index] = value
            else:
                value = self._cache[contribution.source_index]
            unit_weight = (
                len(contributions) == 1 and contribution.weight == 1.0
            )
            if unit_weight:
                term = self._consumer_value(value) if isolate_consumer else value
            else:
                term = value * contribution.weight
            result = term if result is None else result + term
        next_step = index + 1
        keep = (
            set() if next_step >= len(self.plan.schedule)
            else {
                item.source_index
                for item in self.plan.contributions(self.name, next_step)
            }
        )
        retained = {
            source_index: value
            for source_index, value in self._cache.items()
            if source_index in keep
        }
        return result, retained

    @staticmethod
    def _consumer_value(value: Any) -> Any:
        """Detach a unit-weight result from the retained source cache.

        A bundle transaction may be retried after its consumer fails.  The
        consumer must therefore never receive the exact mutable object kept in
        ``_cache``; an in-place write during the failed attempt would otherwise
        alter the forcing observed by the retry while all cursors still report
        the original step.
        """

        if isinstance(value, torch.Tensor):
            return value.clone(memory_format=torch.preserve_format)
        try:
            return copy.deepcopy(value)
        except BaseException as error:
            raise TypeError(
                "a unit-weight forcing sample must be copyable so transaction "
                "retries cannot share mutable reader storage"
            ) from error

    def _commit(self, index: int, retained: dict[int, Any]) -> None:
        """Commit an already staged step using only infallible assignments."""

        self._cache = retained
        self._next_step = index + 1


class ForcingBundle:
    """Atomic multi-source forcing cursor for one model driver.

    ``with bundle.step(step) as inputs`` stages every source first and commits
    their shared cursor only when the surrounding model call returns normally.
    """

    def __init__(
        self,
        plan: ForcingPlan,
        readers: Mapping[str, Callable[[int], Any]],
        *,
        start_step: int = 0,
    ) -> None:
        if not isinstance(plan, ForcingPlan):
            raise TypeError("forcing bundle plan must be a ForcingPlan")
        if not isinstance(readers, Mapping):
            raise TypeError("forcing bundle readers must be a mapping")
        expected = set(plan.sources)
        observed = set(readers)
        if observed != expected:
            raise ValueError(
                "forcing bundle readers must exactly match plan sources: "
                f"missing={sorted(expected - observed)}, "
                f"extra={sorted(observed - expected)}"
            )
        invalid = sorted(
            name for name, reader in readers.items() if not callable(reader)
        )
        if invalid:
            raise TypeError(f"forcing readers must be callable: {invalid}")
        if type(start_step) is not int:
            raise TypeError("forcing bundle start_step must be an exact int")
        if not 0 <= start_step <= len(plan.schedule):
            raise ValueError("forcing bundle start_step is outside the schedule")
        self.plan = plan
        self._streams = MappingProxyType({
            name: ForcingStream(plan, name, readers[name], start_step=start_step)
            for name in plan.sources
        })
        self._next_step = start_step
        self._active = False

    def checkpoint_state(self) -> dict[str, Any]:
        if self._active:
            raise RuntimeError(
                "forcing bundle cannot checkpoint an active step transaction"
            )
        return {
            "fingerprint": self.plan.fingerprint,
            "sources": sorted(self._streams),
            "next_step": self._next_step,
        }

    def restore_checkpoint_state(self, state: Any) -> None:
        if self._active:
            raise RuntimeError(
                "forcing bundle cannot restore during an active step transaction"
            )
        if not isinstance(state, dict) or set(state) != {
            "fingerprint", "sources", "next_step",
        }:
            raise ValueError("forcing bundle checkpoint has an invalid schema")
        if state["fingerprint"] != self.plan.fingerprint:
            raise ValueError("forcing bundle checkpoint plan does not match")
        if state["sources"] != sorted(self._streams):
            raise ValueError("forcing bundle checkpoint sources do not match")
        next_step = state["next_step"]
        if type(next_step) is not int or not 0 <= next_step <= len(
            self.plan.schedule,
        ):
            raise ValueError("forcing bundle checkpoint cursor is outside schedule")
        self._next_step = next_step
        for stream in self._streams.values():
            stream._next_step = next_step
            stream._cache.clear()

    @contextmanager
    def step(self, step: int | SimulationStep):
        """Stage all forcing and commit only after the consumer succeeds."""

        if self._active:
            raise RuntimeError("nested forcing bundle transactions are forbidden")
        index = self.plan._step_index(step)
        if index != self._next_step:
            raise ValueError(
                f"forcing bundle expected model step {self._next_step}, got {index}"
            )
        self._active = True
        try:
            staged: dict[str, Any] = {}
            retained: dict[str, dict[int, Any]] = {}
            for name, stream in self._streams.items():
                staged[name], retained[name] = stream._stage(
                    index, isolate_consumer=True,
                )
            yield MappingProxyType(staged)
        except BaseException:
            raise
        else:
            for name, stream in self._streams.items():
                stream._commit(index, retained[name])
            self._next_step = index + 1
        finally:
            self._active = False

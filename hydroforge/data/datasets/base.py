# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from collections.abc import Iterable, Mapping
from typing import Any, ClassVar, Dict, Iterator, Literal, Optional, Self, Union

import cftime
import numpy as np
import torch
from pydantic import (
    Field,
    PrivateAttr,
    ValidationInfo,
    field_validator,
    model_validator,
)

from hydroforge.contracts.validation import HydroForgeModel, _immutable_dict
from hydroforge.contracts.errors import ResourceCleanupError
from hydroforge.data.numeric import (
    canonical_ids,
    canonical_floating_array,
    immutable_array,
    positive_finite_float64,
)
from hydroforge.contracts.temporal import (
    UpsamplingMethod,
    _DatasetTemporalDomain,
    DateLike,
    SimulationSchedule,
    _require_date,
    canonical_calendar,
    require_calendar,
    timedelta_microseconds,
    _timedelta_quotient_trusted,
)
from hydroforge.data.datasets.chunking import SourceChunk, SourceChunkPlan


_DatasetOperand = Any
_DATASET_INDEX_LENGTH_CONTEXT = "hydroforge_dataset_index_length"
_FORCING_SHARD_CONTEXT = "hydroforge_forcing_shard_context"


class _DatasetIndexQuery(HydroForgeModel):
    """One bounded public Dataset protocol lookup."""

    index: int

    _resolved_index: int = PrivateAttr()

    @model_validator(mode="after")
    def _resolve(self, info: ValidationInfo) -> Self:
        length = (
            info.context.get(_DATASET_INDEX_LENGTH_CONTEXT)
            if isinstance(info.context, Mapping)
            else None
        )
        if type(length) is not int or length < 0:
            raise ValueError("dataset index query requires dataset context")
        index = self.index
        if index < 0:
            index += length
        if not 0 <= index < length:
            raise ValueError(
                f"dataset index must satisfy -{length} <= index < "
                f"{length}; got {self.index}"
            )
        self._resolved_index = index
        return self

    @property
    def resolved_index(self) -> int:
        return self._resolved_index


def _validated_dataset_index(dataset: object, index: int) -> int:
    """Validate one public Dataset index before trusted plan lookup."""

    return _DatasetIndexQuery.model_validate(
        {"index": index},
        context={_DATASET_INDEX_LENGTH_CONTEXT: len(dataset)},
    ).resolved_index


class _ForcingShardRequest(HydroForgeModel):
    """Canonical tensor batch entering a public Dataset sharding method."""

    data: Any

    @model_validator(mode="after")
    def _validate_batch(self, info: ValidationInfo) -> Self:
        context = (
            info.context.get(_FORCING_SHARD_CONTEXT)
            if isinstance(info.context, Mapping)
            else None
        )
        if not isinstance(context, Mapping):
            raise ValueError("forcing shard request requires dataset context")
        allow_sequence = context.get("allow_sequence", False)
        if type(allow_sequence) is not bool:
            raise ValueError("forcing shard sequence context is invalid")
        columns = context.get("columns")
        if type(columns) is not int or columns < 0:
            raise ValueError("forcing shard columns context is invalid")
        dtype = context.get("dtype")
        if dtype is not None and not isinstance(dtype, torch.dtype):
            raise ValueError("forcing shard dtype context is invalid")
        device = context.get("device")
        if device is not None and not isinstance(device, torch.device):
            raise ValueError("forcing shard device context is invalid")

        def canonical(value: Any, *, label: str) -> Any:
            if isinstance(value, Mapping):
                if not value:
                    raise ValueError(f"{label} mapping must not be empty")
                if any(type(name) is not str or not name for name in value):
                    raise ValueError(
                        f"{label} mapping keys must be non-empty exact strings"
                    )
                return {
                    name: canonical(block, label=f"{label}.{name}")
                    for name, block in value.items()
                }
            if isinstance(value, (tuple, list)):
                if not allow_sequence:
                    raise ValueError(f"{label} does not accept a sequence")
                if not value:
                    raise ValueError(f"{label} sequence must not be empty")
                return tuple(
                    canonical(block, label=f"{label}[{index}]")
                    for index, block in enumerate(value)
                )
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"{label} must be a dense torch.Tensor")
            if value.layout != torch.strided:
                raise ValueError(f"{label} must be a dense torch.Tensor")
            if value.ndim not in {2, 3}:
                raise ValueError(
                    f"{label} must have rank 2 or 3; got rank {value.ndim}"
                )
            if value.shape[-1] != columns:
                raise ValueError(
                    f"{label} has {value.shape[-1]} columns; expected {columns}"
                )
            if dtype is not None and value.dtype != dtype:
                raise ValueError(
                    f"{label} has dtype {value.dtype}; expected {dtype}"
                )
            if device is not None and value.device != device:
                raise ValueError(
                    f"{label} is on device {value.device}; expected {device}"
                )
            if (
                (value.is_floating_point() or value.is_complex())
                and not bool(torch.isfinite(value).all().item())
            ):
                raise ValueError(f"{label} contains non-finite values")
            return value

        object.__setattr__(self, "data", canonical(self.data, label="forcing"))
        return self


def _validated_forcing_shard(
    data: Any,
    *,
    columns: int,
    dtype: torch.dtype | None,
    device: torch.device | None,
    allow_sequence: bool = False,
) -> Any:
    """Validate a new public forcing batch before trusted transformation."""

    return _ForcingShardRequest.model_validate(
        {"data": data},
        context={
            _FORCING_SHARD_CONTEXT: {
                "allow_sequence": allow_sequence,
                "columns": columns,
                "dtype": dtype,
                "device": device,
            },
        },
    ).data


class _SourceChunkPayload(HydroForgeModel):
    """Canonical external arrays returned by one storage read."""

    data: Any
    expected_rows: int = Field(strict=True, ge=1, exclude=True)
    clip_negative: bool = Field(strict=True, exclude=True)

    @model_validator(mode="after")
    def _canonicalize(self) -> Self:
        def canonical(value: Any, *, label: str) -> Any:
            if isinstance(value, Mapping):
                if not value:
                    raise ValueError(f"{label} mapping must not be empty")
                if any(type(name) is not str or not name for name in value):
                    raise ValueError(f"{label} keys must be non-empty exact strings")
                return {
                    name: canonical(block, label=f"{label}.{name}")
                    for name, block in value.items()
                }
            if np.ma.isMaskedArray(value):
                mask = np.ma.getmaskarray(value)
                if np.any(mask):
                    value = (
                        value.filled(np.nan)
                        if np.issubdtype(value.dtype, np.floating)
                        else value.astype(np.float64).filled(np.nan)
                    )
                else:
                    value = value.data
            array = np.asarray(value)
            if array.ndim < 1:
                raise ValueError(f"{label} must include a time axis")
            if array.shape[0] != self.expected_rows:
                raise ValueError(
                    f"{label} has {array.shape[0]} rows; expected {self.expected_rows}"
                )
            if array.dtype.kind not in {"f", "i", "u"}:
                raise ValueError(f"{label} must contain real numeric values")
            if np.issubdtype(array.dtype, np.inexact) and not np.isfinite(array).all():
                raise ValueError(f"{label} contains missing or non-finite values")
            owned = np.array(array, order="C", copy=True)
            if self.clip_negative:
                np.maximum(owned, 0, out=owned)
            return owned

        object.__setattr__(
            self,
            "data",
            canonical(self.data, label="source chunk"),
        )
        return self


@dataclass(frozen=True, slots=True)
class _TrustedSourceChunk:
    """Leaf-reader result whose external payload was already validated."""

    data: Any


class _SourceChunkReadRequest(HydroForgeModel):
    """Bind one public chunk identity to the Dataset that owns its plan."""

    chunk: SourceChunk
    temporal_domain: _DatasetTemporalDomain = Field(exclude=True)

    @model_validator(mode="after")
    def _validate_owner(self) -> Self:
        if self.chunk.temporal_domain != self.temporal_domain:
            raise ValueError("source chunk belongs to a different Dataset timeline")
        return self


@dataclass(frozen=True, slots=True)
class _SourceFileIdentity:
    """External file identity captured after Dataset schema validation."""

    device: int
    inode: int
    size: int
    mtime_ns: int

    @classmethod
    def capture(cls, path: Path) -> Self:
        status = path.stat()
        return cls(
            device=status.st_dev,
            inode=status.st_ino,
            size=status.st_size,
            mtime_ns=status.st_mtime_ns,
        )

    def verify(self, path: Path) -> None:
        try:
            observed = type(self).capture(path)
        except OSError as error:
            raise RuntimeError(
                f"Dataset source file {str(path)!r} changed after validation"
            ) from error
        if observed != self:
            raise RuntimeError(
                f"Dataset source file {str(path)!r} changed after validation"
            )


def positive_finite_real(
    value: int | float | np.integer | np.floating,
    *,
    label: str,
) -> float:
    """Return the canonical positive float64 dataset scalar."""

    return positive_finite_float64(value, label=label)


def _close_dataset_tree(root: object, *, scope: str) -> None:
    """Close each unique leaf in one composite dataset ownership tree."""

    pending = [root]
    visited: set[int] = set()
    leaves: list[object] = []
    failures: list[BaseException] = []
    while pending:
        item = pending.pop()
        identity = id(item)
        if identity in visited:
            continue
        visited.add(identity)
        children = getattr(item, "_close_children", None)
        if children is None:
            leaves.append(item)
            continue
        try:
            owned = children()
        except BaseException as error:
            failures.append(error)
            continue
        if owned:
            pending.extend(reversed(owned))
        else:
            leaves.append(item)

    for leaf in leaves:
        try:
            leaf.close()
        except BaseException as error:
            failures.append(error)
    if len(failures) == 1:
        raise failures[0]
    if failures:
        raise ResourceCleanupError(scope, failures)


class _DatasetClassDeclaration(HydroForgeModel):
    """Validated subclass capabilities for physical storage adapters."""

    supports_time_aggregation: bool
    precompressed_source: bool
    integer_output_fields: frozenset[str]

    @field_validator("integer_output_fields", mode="before")
    @classmethod
    def _validate_integer_output_fields(cls, value: Any) -> frozenset[str]:
        if type(value) is not frozenset:
            raise ValueError("integer_output_fields must be an exact frozenset")
        if any(type(name) is not str or not name for name in value):
            raise ValueError(
                "integer_output_fields entries must be non-empty exact strings"
            )
        return value


class AbstractDataset(HydroForgeModel, ABC):
    """
    Custom abstract class that inherits from PyTorch Dataset.
    Defines a common interface for accessing data with distributed support.
    """

    supports_time_aggregation: ClassVar[bool] = False
    precompressed_source: ClassVar[bool] = False

    start_date: DateLike
    end_date: DateLike
    time_interval: timedelta
    model_step: timedelta
    calendar: str | None = None
    spin_up_cycles: int = Field(default=0, strict=True, ge=0)
    spin_up_start_date: DateLike | None = None
    spin_up_end_date: DateLike | None = None
    out_dtype: Literal["float32", "float64"] = "float32"
    chunk_len: int = Field(default=1, strict=True, ge=1)
    clip_negative: bool = Field(default=False, strict=True)
    upsampling: UpsamplingMethod | None = None

    local_indices: np.ndarray | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="Immutable source positions selected by this dataset view",
    )
    desired_catchment_ids: np.ndarray | None = Field(
        default=None,
        exclude=True,
        repr=False,
        description="Immutable target IDs owned by this dataset view",
    )
    _chunk_plan: SourceChunkPlan = PrivateAttr()
    _simulation_schedule: SimulationSchedule = PrivateAttr()
    _temporal_domain: _DatasetTemporalDomain = PrivateAttr()
    _calendar_from_storage_allowed: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def _compile_dataset_identity(self) -> Self:
        """Compile runtime plans from the one canonical temporal identity."""

        temporal_domain = _DatasetTemporalDomain(
            start_date=self.start_date,
            end_date=self.end_date,
            time_interval=self.time_interval,
            calendar=self.calendar,
            spin_up_cycles=self.spin_up_cycles,
            spin_up_start_date=self.spin_up_start_date,
            spin_up_end_date=self.spin_up_end_date,
        )
        self._calendar_from_storage_allowed = (
            temporal_domain.calendar_defaulted
        )
        self._install_temporal_domain(temporal_domain)
        return self

    def _install_temporal_domain(
        self,
        temporal_domain: _DatasetTemporalDomain,
    ) -> None:
        """Install one canonical domain and its derived runtime plans."""

        object.__setattr__(self, "start_date", temporal_domain.start_date)
        object.__setattr__(self, "end_date", temporal_domain.end_date)
        object.__setattr__(self, "calendar", temporal_domain.calendar)
        object.__setattr__(
            self,
            "spin_up_start_date",
            temporal_domain.spin_up_start_date,
        )
        object.__setattr__(
            self,
            "spin_up_end_date",
            temporal_domain.spin_up_end_date,
        )
        self._temporal_domain = temporal_domain

        interval_us = timedelta_microseconds(
            self.time_interval,
            label="dataset time_interval",
        )
        model_step_us = timedelta_microseconds(
            self.model_step,
            label="model_step",
        )
        if model_step_us <= 0:
            raise ValueError("model_step must be positive")
        if model_step_us > interval_us:
            raise ValueError("model_step must not exceed dataset time_interval")
        reuse_count = _timedelta_quotient_trusted(
            self.time_interval,
            self.model_step,
            duration_label="dataset time_interval",
            interval_label="model_step",
        )
        if reuse_count > 1 and self.upsampling not in {"repeat", "distribute"}:
            raise ValueError(
                "upsampling must be explicitly 'repeat' or 'distribute' "
                "when model_step is shorter than dataset time_interval"
            )
        if reuse_count == 1 and self.upsampling is not None:
            raise ValueError(
                "upsampling must be None when model_step equals time_interval"
            )
        self._chunk_plan = SourceChunkPlan(
            temporal_domain=temporal_domain,
            chunk_len=self.chunk_len,
        )
        self._simulation_schedule = SimulationSchedule._from_domain(
            temporal_domain,
            step=self.model_step,
            reuse_count=reuse_count,
        )

    def _adopt_source_calendar(self, calendar: str) -> bool:
        """Bind an omitted calendar to inspected storage metadata once."""

        source = canonical_calendar(calendar)
        if source == self.calendar:
            return False
        if not self._calendar_from_storage_allowed:
            raise ValueError(
                f"forcing files use calendar {source!r}, but the dataset "
                f"declares or implies calendar {self.calendar!r}"
            )
        temporal_domain = _DatasetTemporalDomain(
            start_date=self.start_date,
            end_date=self.end_date,
            time_interval=self.time_interval,
            calendar=source,
            spin_up_cycles=self.spin_up_cycles,
            spin_up_start_date=self.spin_up_start_date,
            spin_up_end_date=self.spin_up_end_date,
        )
        self._calendar_from_storage_allowed = False
        self._install_temporal_domain(temporal_domain)
        return True

    @field_validator("local_indices")
    @classmethod
    def _validate_local_indices(
        cls,
        indices: np.ndarray | None,
    ) -> np.ndarray | None:
        if indices is None:
            return None
        if np.ma.isMaskedArray(indices):
            raise ValueError("dataset local_indices must not be a masked array")
        owned_indices = canonical_ids(
            indices,
            label="dataset local_indices",
        )
        if np.any(owned_indices < 0):
            raise ValueError("dataset local_indices must be nonnegative")
        return immutable_array(owned_indices, order="C")

    @field_validator("desired_catchment_ids")
    @classmethod
    def _validate_desired_catchment_ids(
        cls,
        target_ids: np.ndarray | None,
    ) -> np.ndarray | None:
        if target_ids is None:
            return None
        if np.ma.isMaskedArray(target_ids):
            raise ValueError(
                "dataset desired_catchment_ids must not be a masked array"
            )
        owned_ids = canonical_ids(
            target_ids,
            label="dataset desired_catchment_ids",
        )
        if np.unique(owned_ids).size != owned_ids.size:
            raise ValueError("dataset desired_catchment_ids must be unique")
        return immutable_array(owned_ids, order="C")

    @model_validator(mode="after")
    def _validate_spatial_selection_identity(self) -> Self:
        indices = self.local_indices
        target_ids = self.desired_catchment_ids
        if (indices is None) != (target_ids is None):
            raise ValueError(
                "dataset local_indices and desired_catchment_ids must be "
                "declared together"
            )
        return self

    def _rebuild(self, **updates: Any) -> Self:
        """Construct and validate a new dataset identity from public fields."""

        payload = {name: getattr(self, name) for name in type(self).model_fields}
        payload.update(updates)
        return type(self).model_validate(payload)

    def _dataset_identity_arguments(self) -> dict[str, Any]:
        """Return canonical public fields for an internally created view."""

        return {
            "start_date": self.start_date,
            "end_date": self.end_date,
            "time_interval": self.time_interval,
            "model_step": self.model_step,
            "calendar": self.calendar,
            "spin_up_cycles": self.spin_up_cycles,
            "spin_up_start_date": self.spin_up_start_date,
            "spin_up_end_date": self.spin_up_end_date,
            "out_dtype": self.out_dtype,
            "chunk_len": self.chunk_len,
            "clip_negative": self.clip_negative,
            "upsampling": self.upsampling,
        }

    @property
    def _main_start_time(self):
        """Physical start of the main source support exposed to drivers."""
        return self.start_date

    @property
    def _main_end_time(self):
        """Physical inclusive end of the main source support."""
        return self.end_date

    @property
    def _spin_up_start_time(self):
        """Physical start of the source interval replayed for spin-up."""
        return self.spin_up_start_date

    @property
    def _spin_up_end_time(self):
        """Physical inclusive end of the source interval replayed for spin-up."""
        return self.spin_up_end_date

    @property
    def chunk_plan(self) -> SourceChunkPlan:
        """Return the immutable real-length source-chunk plan."""

        return self._chunk_plan

    @property
    def simulation_schedule(self) -> SimulationSchedule:
        """Return the immutable schedule compiled during initialization."""

        return self._simulation_schedule

    @property
    def _reuse_count(self) -> int:
        """Return the schedule-owned number of model calls per source row."""

        return self._simulation_schedule._reuse_count

    @staticmethod
    def _validate_time_aggregation(method: Optional[str]) -> Optional[str]:
        if method is None:
            return None
        if type(method) is not str:
            raise ValueError("time_aggregation method must be an exact string")
        if method not in ("mean", "max", "min", "sum"):
            raise ValueError(
                f"Unsupported time_aggregation={method!r}; "
                "expected one of: mean, max, min, sum"
            )
        return method

    @classmethod
    def _normalize_time_aggregation(
        cls,
        time_aggregation: str | Mapping[str, str] | None,
    ) -> str | Mapping[str, str] | None:
        if time_aggregation is None:
            return None
        if type(time_aggregation) is str:
            return cls._validate_time_aggregation(time_aggregation)
        if isinstance(time_aggregation, Mapping):
            if not time_aggregation:
                raise ValueError("time_aggregation mapping must not be empty")
            invalid_names = [
                name for name in time_aggregation if type(name) is not str or not name
            ]
            if invalid_names:
                raise ValueError(
                    "time_aggregation names must be non-empty strings; "
                    f"got {invalid_names!r}"
                )
            return _immutable_dict({
                name: cls._validate_time_aggregation(method)
                for name, method in time_aggregation.items()
            })
        raise ValueError("time_aggregation must be None, a string, or a dict")

    def _get_time_aggregation_factor(self, source_time_interval: timedelta) -> int:
        if (
            timedelta_microseconds(
                source_time_interval,
                label="source_time_interval",
            )
            <= 0
        ):
            raise ValueError("source_time_interval must be positive")
        try:
            factor = _timedelta_quotient_trusted(
                self.time_interval,
                source_time_interval,
                duration_label="time_interval",
                interval_label="source_time_interval",
            )
        except ValueError as exc:
            raise ValueError(
                "time_interval must be an exact integer multiple of "
                "source_time_interval for time aggregation"
            ) from exc
        if factor <= 0:
            raise ValueError(
                "time_interval must not be shorter than source_time_interval "
                "for time aggregation"
            )
        return factor

    def _aggregate_time_axis(
        self,
        data: np.ndarray,
        source_time_interval: timedelta,
        method: str,
    ) -> np.ndarray:
        method = self._validate_time_aggregation(method)
        factor = self._get_time_aggregation_factor(source_time_interval)
        if data.shape[0] % factor != 0:
            raise ValueError(
                f"Cannot aggregate {data.shape[0]} source frames into "
                f"windows of {factor} frames"
            )
        grouped = data.reshape((data.shape[0] // factor, factor) + data.shape[1:])
        if method == "mean":
            out = grouped.mean(axis=1, dtype=np.float64)
        elif method == "max":
            out = grouped.max(axis=1)
        elif method == "min":
            out = grouped.min(axis=1)
        elif method == "sum":
            out = grouped.sum(axis=1, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported time_aggregation={method!r}")
        return out

    def _apply_time_aggregation(
        self,
        data: np.ndarray,
        source_time_interval: timedelta,
        time_aggregation: str | Mapping[str, str],
    ) -> np.ndarray | dict[str, np.ndarray]:
        time_aggregation = self._normalize_time_aggregation(time_aggregation)
        if type(time_aggregation) is str:
            return self._aggregate_time_axis(
                data, source_time_interval, time_aggregation
            )
        return {
            name: self._aggregate_time_axis(data, source_time_interval, method)
            for name, method in time_aggregation.items()
        }

    def _finalize_output_data(self, data: Any, *, label: str) -> Any:
        """Cast dataset results only after all numerical transformations."""

        if isinstance(data, Mapping):
            finalized = {
                name: self._finalize_output_data(
                    block,
                    label=f"{label} variable {name!r}",
                )
                for name, block in data.items()
            }
            if all(finalized[name] is block for name, block in data.items()):
                return data
            return finalized
        return canonical_floating_array(
            data,
            dtype=self.out_dtype,
            label=label,
        )

    def _canonical_calculation_data(self, data: Any, *, label: str) -> Any:
        """Own float64 calculation inputs before any arithmetic occurs."""

        if isinstance(data, Mapping):
            return {
                name: self._canonical_calculation_data(
                    block,
                    label=f"{label} variable {name!r}",
                )
                for name, block in data.items()
            }
        return canonical_floating_array(
            data,
            dtype="float64",
            label=label,
        )

    def _require_calendar_datetime(
        self,
        value: Union[datetime, cftime.datetime],
        *,
        label: str,
    ) -> Union[datetime, cftime.datetime]:
        """Validate one storage timestamp without changing its calendar."""

        _require_date(value, label=label)
        require_calendar(value, self.calendar, label=label)
        if type(value) is not type(self.start_date):
            raise ValueError(
                f"{label} must use the same datetime representation as "
                "dataset start_date"
            )
        return value

    @staticmethod
    def _as_nan_array(data: np.ndarray) -> np.ndarray:
        """Convert NetCDF masked values to NaN while preserving normal values."""
        if isinstance(data, np.ma.MaskedArray):
            mask = np.ma.getmaskarray(data)
            if np.any(mask):
                if np.issubdtype(data.dtype, np.floating):
                    return np.asarray(data.filled(np.nan))
                return np.asarray(data.astype(np.float64).filled(np.nan))
            return np.asarray(data.data)
        return np.asarray(data)

    def _apply_upsampling_policy(self, data: Any) -> Any:
        if self.upsampling != "distribute":
            return data
        if isinstance(data, Mapping):
            return {
                name: self._apply_upsampling_policy(value)
                for name, value in data.items()
            }
        calculation = self._canonical_calculation_data(
            data,
            label="distributed upsampling input",
        )
        distributed = calculation / self._reuse_count
        return self._finalize_output_data(
            distributed,
            label="distributed upsampling output",
        )

    def _validate_source_calendar(self, calendar: str) -> None:
        """Accept inferred storage metadata or reject a real conflict."""

        source = canonical_calendar(calendar)
        if source != self.calendar:
            self._adopt_source_calendar(source)

    def _forcing_blocks(self, chunk: Any) -> Iterator[tuple[Any, int]]:
        """Yield each source item once with its exact model-call reuse count."""

        if isinstance(chunk, Mapping):
            length = len(next(iter(chunk.values())))
            for index in range(length):
                yield (
                    {name: value[index] for name, value in chunk.items()},
                    self._reuse_count,
                )
            return
        for item in chunk:
            yield item, self._reuse_count

    @property
    def _num_spin_up_chunks(self) -> int:
        return self._chunk_plan.num_spinup_chunks

    def read_chunk(
        self,
        chunk: SourceChunk,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Read exactly one immutable request from this dataset's source plan."""

        request = _SourceChunkReadRequest(
            chunk=chunk,
            temporal_domain=self._temporal_domain,
        )
        return self._read_chunk_trusted(request.chunk)

    def _read_chunk_trusted(
        self,
        chunk: SourceChunk,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Read one framework-produced chunk without revalidating identity."""

        return self._accept_read_chunk(
            self._read_chunk(chunk),
            chunk,
        )

    def _accept_read_chunk(self, data: Any, chunk: SourceChunk) -> Any:
        """Accept a trusted composite result without validating it again."""

        del chunk
        return data

    def get_chunk(
        self,
        chunk: SourceChunk,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Read and normalize one exact source request for consumption."""

        request = _SourceChunkReadRequest(
            chunk=chunk,
            temporal_domain=self._temporal_domain,
        )
        return self._get_chunk_trusted(request.chunk)

    def _get_chunk_trusted(
        self,
        chunk: SourceChunk,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Prepare one framework-produced chunk without revalidation."""

        compressed = self.local_indices is not None or self.precompressed_source
        data = self._read_chunk_trusted(chunk)
        integer_fields = getattr(
            type(self), "integer_output_fields", frozenset(),
        )
        if isinstance(data, dict):
            missing_integer_fields = integer_fields.difference(data)
            if missing_integer_fields:
                raise ValueError(
                    "integer_output_fields are absent from the source chunk: "
                    f"{sorted(missing_integer_fields)!r}"
                )
            return {
                name: (
                    self._prepare_integer_output_array(block, label=name)
                    if name in integer_fields
                    else self._prepare_chunk_array(block, compressed)
                )
                for name, block in data.items()
            }
        if integer_fields:
            raise ValueError(
                "integer_output_fields require source chunks to be mappings"
            )
        return self._prepare_chunk_array(data, compressed)

    @staticmethod
    def _prepare_integer_output_array(
        data: Any,
        *,
        label: str,
    ) -> np.ndarray:
        """Canonicalize one declared integer output without float conversion."""

        if np.ma.isMaskedArray(data):
            raise ValueError(
                f"prepared integer forcing field {label!r} must not be masked"
            )
        array = np.asarray(data)
        if array.ndim < 1:
            raise ValueError(
                f"prepared integer forcing field {label!r} must include a "
                "time axis"
            )
        if array.dtype.kind not in {"i", "u"}:
            raise ValueError(
                f"prepared integer forcing field {label!r} must contain "
                "integers"
            )
        if (
            array.dtype.kind == "u"
            and array.size
            and int(array.max()) > np.iinfo(np.int64).max
        ):
            raise ValueError(
                f"prepared integer forcing field {label!r} contains a value "
                "outside int64 range"
            )
        return np.array(array, dtype=np.int64, order="C", copy=True)

    @abstractmethod
    def _read_chunk(
        self,
        chunk: SourceChunk,
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Interpret one validated temporal request through source storage.

        Implementations interpret the same temporal request through their own
        storage layout. Returned arrays retain ``chunk.length`` real rows and
        are never padded.
        """

        ...

    @property
    def num_main_source_steps(self) -> int:
        return self._temporal_domain.count

    @property
    def _num_spin_up_source_steps_per_cycle(self) -> int:
        return self._chunk_plan.spinup_source_count_per_cycle

    @abstractmethod
    def close(self) -> None:
        """
        Close any open resources or files. Implementations must be idempotent.
        """

    def _close_children(self) -> tuple[object, ...]:
        """Return directly owned datasets; leaves own no child datasets."""

        return ()

    def _combine(self, other, operation, reverse=False):
        from hydroforge.data.datasets.expression import DatasetExpression

        is_dataset = isinstance(other, (AbstractDataset, DatasetExpression))
        is_scalar = (
            isinstance(other, (int, float, np.integer, np.floating))
            and not isinstance(other, (bool, np.bool_))
            and np.isfinite(other)
        )

        if not (is_dataset or is_scalar):
            return NotImplemented

        left, right = (other, self) if reverse else (self, other)
        return DatasetExpression(
            left=left,
            operation=operation,
            right=right,
        )

    def __getitem__(self, idx: int) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """
        Fetch one chunk (T <= chunk_len) starting at chunk index `idx`.

        The final chunk retains its real time length instead of being padded.
        """
        chunk = self._chunk_plan._at_trusted(
            _validated_dataset_index(self, idx),
        )
        return self._get_chunk_trusted(chunk)

    def _prepare_chunk_array(
        self,
        data: np.ndarray,
        compressed: bool,
    ) -> np.ndarray:
        del compressed
        data = self._apply_upsampling_policy(data)
        return self._finalize_output_data(
            data,
            label="prepared forcing chunk",
        )

    def __len__(self) -> int:
        return len(self._chunk_plan)

    def __add__(self, other: _DatasetOperand):
        return self._combine(other, "add")

    def __radd__(self, other: _DatasetOperand):
        return self._combine(other, "add", reverse=True)

    def __sub__(self, other: _DatasetOperand):
        return self._combine(other, "sub")

    def __rsub__(self, other: _DatasetOperand):
        return self._combine(other, "sub", reverse=True)

    def __mul__(self, other: _DatasetOperand):
        return self._combine(other, "mul")

    def __rmul__(self, other: _DatasetOperand):
        return self._combine(other, "mul", reverse=True)

    def __truediv__(self, other: _DatasetOperand):
        return self._combine(other, "div")

    def __rtruediv__(self, other: _DatasetOperand):
        return self._combine(other, "div", reverse=True)


class SourceDataset(AbstractDataset, ABC):
    """Validated declarative fields shared by physical storage adapters."""

    integer_output_fields: ClassVar[frozenset[str]] = frozenset()

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        declaration = _DatasetClassDeclaration(
            supports_time_aggregation=cls.supports_time_aggregation,
            precompressed_source=cls.precompressed_source,
            integer_output_fields=cls.integer_output_fields,
        )
        cls.supports_time_aggregation = declaration.supports_time_aggregation
        cls.precompressed_source = declaration.precompressed_source
        cls.integer_output_fields = declaration.integer_output_fields

    _source_file_identities: Mapping[Path, _SourceFileIdentity] = PrivateAttr(
        default_factory=dict,
    )

    def _accept_read_chunk(self, data: Any, chunk: SourceChunk) -> Any:
        """Validate one raw leaf result, or unwrap an already validated read."""

        if isinstance(data, _TrustedSourceChunk):
            return data.data
        if isinstance(data, Mapping) and self.integer_output_fields:
            if not data:
                raise ValueError("source chunk mapping must not be empty")
            if any(type(name) is not str or not name for name in data):
                raise ValueError(
                    "source chunk mapping keys must be non-empty exact strings"
                )
            return {
                name: _SourceChunkPayload(
                    data=block,
                    expected_rows=chunk.length,
                    clip_negative=(
                        False
                        if name in self.integer_output_fields
                        else self.clip_negative
                    ),
                ).data
                for name, block in data.items()
            }
        return _SourceChunkPayload(
            data=data,
            expected_rows=chunk.length,
            clip_negative=self.clip_negative,
        ).data

    @staticmethod
    def _canonical_source_path(path: str | Path) -> Path:
        return Path(path).absolute()

    def _record_source_files(self, paths: Iterable[str | Path]) -> None:
        """Freeze every schema-inspected source file identity."""

        canonical = tuple(
            dict.fromkeys(self._canonical_source_path(path) for path in paths)
        )
        self._source_file_identities = {
            path: _SourceFileIdentity.capture(path) for path in canonical
        }

    def _checked_source_path(self, path: str | Path) -> Path:
        """Verify external identity immediately before one runtime read."""

        canonical = self._canonical_source_path(path)
        self._source_file_identities[canonical].verify(canonical)
        return canonical

    def _verify_source_path(self, path: str | Path) -> None:
        """Verify external identity immediately after one runtime read."""

        canonical = self._canonical_source_path(path)
        self._source_file_identities[canonical].verify(canonical)

    def _validate_local_index_extent(self, size: int, *, label: str) -> None:
        """Validate this immutable spatial view against source storage."""

        if (
            self.local_indices is not None
            and self.local_indices.size
            and int(self.local_indices.max()) >= size
        ):
            raise ValueError(f"dataset local_indices exceed {label} size {size}")

    @property
    @abstractmethod
    def data_size(self) -> int:
        """Number of spatial values produced by each source step."""

    @abstractmethod
    def get_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the coordinate identity associated with ``data_size``."""

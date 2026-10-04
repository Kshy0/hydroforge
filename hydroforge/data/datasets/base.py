# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""The forcing-dataset contract and the composites its arithmetic builds.

A :class:`ForcingDataset` declares a timeline and compiles it once into a
:class:`~hydroforge.data.datasets.plan.DatasetPlan`.  Declared fields keep the
caller's values; resolved values (calendar, chunk length, normalized dates)
live only in the plan.  Reads run one value pipeline in the reading process,
normally a DataLoader worker.  Spatial selection never changes a dataset:
:meth:`ForcingDataset.build_local_mapping` returns a view that shares the plan and
files and reads only the mapped source cells.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping
from datetime import timedelta
from operator import add, mul, sub, truediv
from pathlib import Path
from typing import Annotated, Any, ClassVar, Literal, Self

import numpy as np
import torch
from pydantic import (
    AfterValidator,
    BeforeValidator,
    Field,
    PrivateAttr,
    model_validator,
    validate_call,
)

from hydroforge.core.arrays import UniqueIds, canonical_floating_array
from hydroforge.core.devices import devices_match
from hydroforge.core.errors import cleanup_on_exit
from hydroforge.core.time import DateLike
from hydroforge.core.validation import HydroForgeModel
from hydroforge.data.datasets.plan import (
    DatasetPlan,
    SourceChunk,
    SourceChunkPlan,
    TemporalDomain,
    UpsamplingMethod,
    compile_cadence,
    plan_index,
)
from hydroforge.data.datasets.space import GridSpace, PointSpace
from hydroforge.data.datasets.values import (
    MissingPolicy,
    bounded,
    finalize,
    ingest,
    ingest_integer,
)
from hydroforge.mapping.table import MappingTable

# Source identities are absolute paths captured at construction, so the
# storage directory is bound then as well; a later working-directory change
# must not redirect a relative declaration.
SourceDirectory = Annotated[
    Path,
    Field(strict=False),
    AfterValidator(Path.absolute),
]
TorchDevice = Annotated[torch.device, BeforeValidator(torch.device)]
_TORCH_DTYPES = {"float32": torch.float32, "float64": torch.float64}


def _check_forcing_tensors(
    value: Any,
    *,
    columns: int,
    dtype: torch.dtype,
    device: torch.device | None,
    sequence: bool,
    label: str,
    integer_fields: frozenset[str] = frozenset(),
) -> None:
    """Check one forcing batch, with optional top-level int64 field names."""

    if integer_fields and not isinstance(value, Mapping):
        raise ValueError(f"{label} integer_output_fields require a mapping")

    if isinstance(value, Mapping):
        if not value or any(type(name) is not str or not name for name in value):
            raise ValueError(f"{label} keys must be non-empty exact strings")
        absent = integer_fields.difference(value)
        if absent:
            raise ValueError(
                f"{label} is missing integer_output_fields: {sorted(absent)!r}"
            )
        for name, block in value.items():
            _check_forcing_tensors(
                block,
                columns=columns,
                dtype=torch.int64 if name in integer_fields else dtype,
                device=device,
                sequence=sequence,
                label=f"{label}.{name}",
            )
        return
    if isinstance(value, (tuple, list)):
        if not sequence:
            raise ValueError(f"{label} does not accept a sequence")
        if not value:
            raise ValueError(f"{label} sequence must not be empty")
        for index, block in enumerate(value):
            _check_forcing_tensors(
                block,
                columns=columns,
                dtype=dtype,
                device=device,
                sequence=sequence,
                label=f"{label}[{index}]",
            )
        return
    if not isinstance(value, torch.Tensor) or value.layout != torch.strided:
        raise ValueError(f"{label} must be a dense torch.Tensor")
    if value.ndim not in {2, 3}:
        raise ValueError(f"{label} must have rank 2 or 3; got rank {value.ndim}")
    if value.shape[-1] != columns:
        raise ValueError(f"{label} has {value.shape[-1]} columns; expected {columns}")
    if value.dtype != dtype:
        raise ValueError(f"{label} has dtype {value.dtype}; expected {dtype}")
    if device is not None and not devices_match(value.device, device):
        raise ValueError(f"{label} is on device {value.device}; expected {device}")


def _mapping_layout(device: torch.device) -> torch.layout:
    """Sparse layout of :meth:`ForcingDataset.build_local_mapping` tensors on ``device``.

    CUDA uses CSR, which cuSPARSE multiplies without per-call conversion.
    Other devices use coalesced COO, whose product sums each target's
    sources in index order; the CPU CSR kernel sums in a different order.
    """

    return torch.sparse_csr if device.type == "cuda" else torch.sparse_coo


def _mapped_forcing(value: Any, mapping: torch.Tensor) -> Any:
    """Project ``(..., sources)`` batches onto ``(..., targets)``."""

    if isinstance(value, Mapping):
        return {name: _mapped_forcing(block, mapping) for name, block in value.items()}
    leading = value.shape[:-1]
    flat = value.reshape(math.prod(leading), value.shape[-1])
    return (mapping @ flat.T).T.contiguous().view(*leading, mapping.shape[0])


class ForcingDataset(HydroForgeModel, ABC):
    """A declared forcing source read chunk by chunk on one compiled timeline.

    Subclasses provide :attr:`space` and either :meth:`read_storage` (raw
    arrays that the base class checks and converts) or their own read
    pipeline.  ``missing`` decides whether masked and NaN values are zeroed
    or rejected; infinities are always rejected.  :meth:`shard_forcing` checks
    only structure, shape, dtype and device: values were checked when read.
    """

    integer_output_fields: ClassVar[frozenset[str]] = frozenset()

    start_date: DateLike
    end_date: DateLike
    time_interval: timedelta
    model_step: timedelta
    calendar: str | None = None
    spin_up_cycles: int = Field(default=0, ge=0)
    spin_up_start_date: DateLike | None = None
    spin_up_end_date: DateLike | None = None
    out_dtype: Literal["float32", "float64"] = "float32"
    chunk_len: int = Field(default=1, ge=1)
    clip_negative: bool = False
    missing: MissingPolicy = "error"
    upsampling: UpsamplingMethod | None = None

    _plan: DatasetPlan = PrivateAttr()
    _target_ids: np.ndarray | None = PrivateAttr(default=None)

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        fields = cls.integer_output_fields
        if type(fields) is not frozenset or any(
            type(name) is not str or not name for name in fields
        ):
            raise TypeError(
                "integer_output_fields must be a frozenset of non-empty strings"
            )

    @model_validator(mode="after")
    def _compile(self) -> Self:
        # Pydantic re-runs after-validators on an instance embedded in another
        # model; the plan of a constructed dataset is final.
        if "_plan" not in self.__pydantic_private__:
            compile_cadence(self.time_interval, self.model_step, self.upsampling)
            self._plan = self._compile_plan(self._declared_domain())
        return self

    def _declared_domain(self, calendar: str | None = None) -> TemporalDomain:
        """The domain of the declared dates, optionally in a storage calendar."""

        return TemporalDomain.declare(
            start_date=self.start_date,
            end_date=self.end_date,
            time_interval=self.time_interval,
            calendar=self.calendar if calendar is None else calendar,
            spin_up_cycles=self.spin_up_cycles,
            spin_up_start_date=self.spin_up_start_date,
            spin_up_end_date=self.spin_up_end_date,
        )

    def _compile_plan(self, domain: TemporalDomain) -> DatasetPlan:
        """Bind storage to the declared domain; storage leaves override this."""

        return self._planned(domain, self.chunk_len)

    def _planned(self, domain: TemporalDomain, chunk_len: int) -> DatasetPlan:
        return DatasetPlan.compile(
            domain,
            chunk_len=chunk_len,
            model_step=self.model_step,
            upsampling=self.upsampling,
        )

    # ------------------------------------------------------------------
    # Compiled identity
    # ------------------------------------------------------------------
    @property
    def chunk_plan(self) -> SourceChunkPlan:
        """The real-length source chunks, spin-up cycles first."""

        return self._plan.chunk_plan

    @property
    def simulation_schedule(self):
        """The model schedule, including the resolved calendar."""

        return self._plan.schedule

    @property
    def num_main_source_steps(self) -> int:
        return self._plan.domain.count

    @property
    @abstractmethod
    def space(self) -> GridSpace | PointSpace:
        """The spatial identity of each value row (with this view's selection)."""

    @property
    def data_size(self) -> int:
        """Number of values in each source row."""

        return self.space.size

    @property
    def target_ids(self) -> np.ndarray | None:
        """Mapping targets of a :meth:`build_local_mapping` view, else ``None``."""

        return self._target_ids

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------
    def read(self, chunk: SourceChunk) -> np.ndarray | dict[str, np.ndarray]:
        """Read one chunk of this dataset's plan at model cadence.

        Views share their source's plan, so its chunks are accepted by both;
        any other chunk object is rejected.
        """

        if not isinstance(chunk, SourceChunk):
            raise TypeError("read() requires a SourceChunk of this chunk_plan")
        chunks = self._plan.chunk_plan.chunks
        if not (0 <= chunk.index < len(chunks)) or chunks[chunk.index] is not chunk:
            raise ValueError(
                "source chunk does not belong to this dataset's chunk plan"
            )
        return self._read_at(chunk.index)

    def __getitem__(self, index: int) -> np.ndarray | dict[str, np.ndarray]:
        """Read chunk ``index`` (the DataLoader entry point)."""

        return self._read_at(plan_index(index, len(self), label="dataset"))

    def __len__(self) -> int:
        return len(self._plan.chunk_plan)

    def _read_at(self, index: int) -> np.ndarray | dict[str, np.ndarray]:
        """Chunk ``index`` at model cadence (``distribute`` applied)."""

        return self._distributed(self._read_source(index))

    def _read_source(self, index: int) -> np.ndarray | dict[str, np.ndarray]:
        """Chunk ``index`` at source cadence, checked and in ``out_dtype``."""

        chunk = self._plan.chunk_plan.chunks[index]
        return self._converted(self.read_storage(chunk), chunk.length)

    def read_storage(self, chunk: SourceChunk) -> Any:
        """Return the raw values of ``chunk``: an array or a name→array mapping.

        Arrays have ``chunk.length`` rows.  The caller receives them, so return
        a copy of any array the dataset keeps.
        """

        raise NotImplementedError(f"{type(self).__name__} does not read storage")

    def _converted(self, raw: Any, rows: int) -> np.ndarray | dict[str, np.ndarray]:
        fields = type(self).integer_output_fields
        if not isinstance(raw, Mapping):
            if fields:
                raise ValueError(
                    "integer_output_fields require source chunks to be mappings"
                )
            return self._convert(self._ingest(raw, rows, label="source chunk"))
        if not raw or any(type(name) is not str or not name for name in raw):
            raise ValueError(
                "source chunk mapping keys must be non-empty exact strings"
            )
        absent = fields.difference(raw)
        if absent:
            raise ValueError(
                "integer_output_fields are absent from the source chunk: "
                f"{sorted(absent)!r}"
            )
        return {
            name: (
                ingest_integer(block, rows=rows, label=f"source chunk {name!r}")
                if name in fields
                else self._convert(
                    self._ingest(block, rows, label=f"source chunk {name!r}")
                )
            )
            for name, block in raw.items()
        }

    def _ingest(self, raw: Any, rows: int, *, label: str) -> np.ndarray:
        return ingest(
            raw,
            rows=rows,
            missing=self.missing,
            clip_negative=self.clip_negative,
            label=label,
        )

    def _convert(self, values: np.ndarray) -> np.ndarray | dict[str, np.ndarray]:
        """Narrow one ingested read to ``out_dtype``; storage leaves override."""

        return finalize(
            values,
            out_dtype=self.out_dtype,
            checked=not (
                values.dtype.kind == "f" and bounded(values.dtype, self.out_dtype)
            ),
            label="prepared forcing chunk",
        )

    def _distributed(self, data: Any) -> Any:
        """Split each source value evenly over its model steps if requested."""

        if self.upsampling != "distribute":
            return data
        if isinstance(data, Mapping):
            return {
                name: (
                    block
                    if name in type(self).integer_output_fields
                    else self._distributed(block)
                )
                for name, block in data.items()
            }
        return finalize(
            np.asarray(data, dtype=np.float64) / self._plan.reuse_count,
            out_dtype=self.out_dtype,
            checked=False,
            label="distributed upsampling output",
        )

    def close(self) -> None:
        """Close process-local resources; idempotent (later reads reopen)."""

    def _first_frame_missing(self) -> np.ndarray | None:
        """``(Y, X)`` missing-value mask of the first source frame, if readable."""

        return None

    # ------------------------------------------------------------------
    # Views and sharding
    # ------------------------------------------------------------------
    def _view(self, fields: Mapping[str, Any] | None = None, /, **private: Any) -> Self:
        """A dataset sharing this plan and storage with updated view state."""

        view = type(self).model_construct(**(self.__dict__ | dict(fields or {})))
        object.__setattr__(
            view, "__pydantic_private__", self.__pydantic_private__ | private
        )
        return view

    def replaced(self, **fields: Any) -> Self:
        """Return a newly validated dataset declaring ``fields`` instead."""

        return type(self).model_validate(self.__dict__ | fields)

    @validate_call(config=HydroForgeModel.model_config)
    def build_local_mapping(
        self,
        mapping_file: Annotated[Path, Field(strict=False)],
        target_ids: UniqueIds | None = None,
        *,
        device: TorchDevice = torch.device("cpu"),
        precision: Literal["float32", "float64"] = "float32",
    ) -> tuple[Self, torch.Tensor]:
        """Return a view reading the mapped source cells, and its mapping.

        The mapping tensor is the sparse ``(targets, active sources)`` matrix
        of the saved table restricted to ``target_ids`` (all targets when
        ``None``): CSR on CUDA, coalesced COO on other devices.  This dataset
        is not changed; mapping a view again selects from the full source
        grid.
        """

        space = self.space
        if not isinstance(space, GridSpace):
            raise TypeError(f"{type(self).__name__} has no source grid to map")
        table = MappingTable.load(mapping_file)
        table.require_source_grid(space.longitude, space.latitude)
        local = table.local(target_ids)
        tensor = local.to_torch(
            device=device,
            dtype=_TORCH_DTYPES[precision],
            layout=_mapping_layout(device),
        )
        return self._mapped(local.source_indices, local.target_ids), tensor

    def _mapped(self, source_indices: np.ndarray, target_ids: np.ndarray) -> Self:
        return self._view(
            _space=self.space.select(source_indices),
            _target_ids=target_ids,
        )

    def shard_forcing(self, chunk: Any, mapping: torch.Tensor | None = None) -> Any:
        """Put one device batch of this dataset's reads on its targets.

        Mapped grid views multiply by the tensor from :meth:`build_local_mapping`;
        point datasets take no mapping and return the batch.  Only structure,
        shape, dtype and device are checked, without device synchronization.
        """

        self._check_forcing(chunk, mapping, label="forcing")
        return chunk if mapping is None else _mapped_forcing(chunk, mapping)

    def _check_forcing(
        self, chunk: Any, mapping: torch.Tensor | None, *, label: str
    ) -> None:
        space = self.space
        if isinstance(space, PointSpace):
            if mapping is not None:
                raise ValueError("point datasets are sharded without a mapping")
            _check_forcing_tensors(
                chunk,
                columns=space.size,
                dtype=_TORCH_DTYPES[self.out_dtype],
                device=None,
                sequence=True,
                label=label,
                integer_fields=type(self).integer_output_fields,
            )
            return
        self._require_mapping(mapping, caller="shard_forcing()")
        _check_forcing_tensors(
            chunk,
            columns=space.size,
            dtype=mapping.dtype,
            device=mapping.device,
            sequence=False,
            label=label,
        )

    def _require_mapping(self, mapping: Any, *, caller: str) -> np.ndarray:
        """Target IDs of this grid view once ``mapping`` is its mapping tensor."""

        targets = self.target_ids
        if not isinstance(self.space, GridSpace) or targets is None:
            raise ValueError(
                f"{caller} requires a grid dataset view returned by build_local_mapping()"
            )
        expected = (targets.size, self.data_size)
        if (
            not isinstance(mapping, torch.Tensor)
            or mapping.layout != _mapping_layout(mapping.device)
            or tuple(mapping.shape) != expected
        ):
            observed = (
                f"{mapping.layout} with shape {tuple(mapping.shape)}"
                if isinstance(mapping, torch.Tensor)
                else type(mapping).__name__
            )
            raise ValueError(
                "mapping must be the sparse tensor build_local_mapping() returns on its "
                f"device, with shape {expected}; got {observed}"
            )
        return targets

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------
    def _combine(self, other: Any, operation: str, *, reverse: bool = False) -> Any:
        if not (isinstance(other, ForcingDataset) or _is_scalar(other)):
            return NotImplemented
        left, right = (other, self) if reverse else (self, other)
        return DatasetExpression(left=left, operation=operation, right=right)

    def __add__(self, other: Any):
        return self._combine(other, "add")

    def __radd__(self, other: Any):
        return self._combine(other, "add", reverse=True)

    def __sub__(self, other: Any):
        return self._combine(other, "sub")

    def __rsub__(self, other: Any):
        return self._combine(other, "sub", reverse=True)

    def __mul__(self, other: Any):
        return self._combine(other, "mul")

    def __rmul__(self, other: Any):
        return self._combine(other, "mul", reverse=True)

    def __truediv__(self, other: Any):
        return self._combine(other, "div")

    def __rtruediv__(self, other: Any):
        return self._combine(other, "div", reverse=True)


class CompositeDataset(ForcingDataset, ABC):
    """A dataset computed from children that share one plan and one space.

    The declaration mirrors the first child; the plan is that child's plan
    object, not a recompilation.  Children keep their own storage and
    missing-value policies.
    """

    # Declarations mirror the reference child, whose chunk length may be
    # planned from storage.
    chunk_len: int | None = Field(default=None, ge=1)

    _reference: ForcingDataset = PrivateAttr()

    @classmethod
    @abstractmethod
    def _declared_reference(cls, value: Mapping[str, Any]) -> Any:
        """The first child in a raw declaration, if it is a dataset."""

    @abstractmethod
    def _children(self) -> tuple[tuple[str, ForcingDataset], ...]:
        """Labelled child datasets, reference first."""

    @model_validator(mode="before")
    @classmethod
    def _mirror_declaration(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        reference = cls._declared_reference(value)
        if not isinstance(reference, ForcingDataset):
            return value
        payload = dict(value)
        for name in ForcingDataset.model_fields:
            expected = getattr(reference, name)
            if name in payload and payload[name] != expected:
                raise ValueError(f"{cls.__name__} {name} must match its children")
            payload[name] = expected
        return payload

    @model_validator(mode="after")
    def _compile(self) -> Self:
        if "_plan" in self.__pydantic_private__:
            return self
        (_name, reference), *others = self._children()
        for label, child in others:
            self._require_compatible(reference, child, label=label)
        self._reference = reference
        self._plan = reference._plan
        return self

    def _require_compatible(
        self, reference: ForcingDataset, child: ForcingDataset, *, label: str
    ) -> None:
        reference._plan.require_equivalent(child._plan, label=label)
        space, other = reference.space, child.space
        if type(space) is not type(other):
            raise ValueError(f"{label} cannot mix grid and point sources")
        space.require_compatible(other, label=label)
        windows = []
        for dataset in (reference, child):
            while isinstance(dataset, CompositeDataset):
                dataset = dataset._reference
            starts = getattr(dataset, "window_starts", None)
            windows.append(
                (
                    getattr(dataset, "window_length", None)
                    if starts is not None
                    else None,
                    starts,
                )
            )
        if windows[0][0] != windows[1][0] or not np.array_equal(
            windows[0][1], windows[1][1]
        ):
            raise ValueError(f"{label} must share the same sampling windows")

    def __len__(self) -> int:
        return len(self._reference)

    def __getitem__(self, index: int) -> Any:
        return self._read_item(plan_index(index, len(self), label="dataset"))

    @abstractmethod
    def _read_item(self, index: int) -> Any:
        """Read children through their public chunk or window item protocol."""

    @property
    def space(self) -> GridSpace | PointSpace:
        return self._reference.space

    @property
    def target_ids(self) -> np.ndarray | None:
        return self._reference.target_ids

    def _unique_children(self) -> tuple[ForcingDataset, ...]:
        return tuple({id(child): child for _name, child in self._children()}.values())

    def close(self) -> None:
        with cleanup_on_exit(
            f"{type(self).__name__} resources",
            [child.close for child in self._unique_children()],
        ):
            pass


_OPERATIONS: dict[str, Callable[[Any, Any], Any]] = {
    "add": add,
    "sub": sub,
    "mul": mul,
    "div": truediv,
}


def _is_scalar(value: Any) -> bool:
    return (
        isinstance(value, (int, float, np.integer, np.floating))
        and not isinstance(value, (bool, np.bool_))
        and bool(np.isfinite(value))
    )


def _operand(value: Any) -> Any:
    if isinstance(value, ForcingDataset) or _is_scalar(value):
        return value
    raise ValueError(
        "dataset expression operands must be datasets or finite real numbers"
    )


_Operand = Annotated[Any, AfterValidator(_operand)]


def _evaluate_expression(
    operation: str,
    left: Any,
    right: Any,
    *,
    left_is_scalar: bool,
    right_is_scalar: bool,
    out_dtype: str,
) -> Any:
    left_mapping = isinstance(left, Mapping)
    right_mapping = isinstance(right, Mapping)
    if left_mapping or right_mapping:
        if left_mapping and right_mapping:
            if set(left) != set(right):
                raise ValueError(
                    "dataset expression mappings must have identical variable names"
                )
            pairs = {name: (left[name], right[name]) for name in left}
        elif left_mapping and right_is_scalar:
            pairs = {name: (block, right) for name, block in left.items()}
        elif right_mapping and left_is_scalar:
            pairs = {name: (left, block) for name, block in right.items()}
        else:
            raise TypeError(
                "dataset expression operands must return matching mappings or "
                "numeric arrays"
            )
        return {
            name: _evaluate_expression(
                operation,
                first,
                second,
                left_is_scalar=left_is_scalar and not left_mapping,
                right_is_scalar=right_is_scalar and not right_mapping,
                out_dtype=out_dtype,
            )
            for name, (first, second) in pairs.items()
        }

    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if (
        not left_is_scalar
        and not right_is_scalar
        and left_array.shape != right_array.shape
    ):
        raise ValueError(
            "dataset expression operands must return identical shapes; got "
            f"{left_array.shape} and {right_array.shape}"
        )
    # Evaluate in the declared output dtype: arithmetic in an operand's own
    # integer or narrower dtype could overflow before the result is checked.
    left_array = canonical_floating_array(
        left_array,
        dtype=out_dtype,
        label="left dataset expression operand",
    )
    right_array = canonical_floating_array(
        right_array,
        dtype=out_dtype,
        label="right dataset expression operand",
    )
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        result = _OPERATIONS[operation](left_array, right_array)
    try:
        return canonical_floating_array(
            result,
            dtype=out_dtype,
            label="dataset expression result",
        )
    except ValueError:
        if np.isinf(result).any() and not (
            operation == "div" and np.any(right_array == 0)
        ):
            if out_dtype == "float32":
                raise OverflowError(
                    "dataset expression result contains values outside float32 range"
                ) from None
            raise OverflowError(
                "dataset expression result overflowed float64"
            ) from None
        raise


class DatasetExpression(CompositeDataset):
    """A lazy arithmetic expression of compatible datasets and scalars.

    Each dataset operand is read once per chunk and applies its own upsampling
    before the arithmetic, evaluated in the reference ``out_dtype``.
    """

    left: _Operand
    operation: Literal["add", "sub", "mul", "div"]
    right: _Operand

    _left: Any = PrivateAttr()
    _right: Any = PrivateAttr()

    @classmethod
    def _declared_reference(cls, value: Mapping[str, Any]) -> Any:
        return next(
            (
                operand
                for operand in (value.get("left"), value.get("right"))
                if isinstance(operand, ForcingDataset)
            ),
            None,
        )

    def _children(self) -> tuple[tuple[str, ForcingDataset], ...]:
        children = tuple(
            operand
            for operand in (self.left, self.right)
            if isinstance(operand, ForcingDataset)
        )
        return tuple(
            (f"dataset operand {position}", child)
            for position, child in enumerate(children)
        )

    @model_validator(mode="after")
    def _compile(self) -> Self:
        if "_plan" in self.__pydantic_private__:
            return self
        if not self._children():
            raise ValueError("a dataset expression requires at least one dataset")
        super()._compile()
        self._left, self._right = (
            operand
            if isinstance(operand, ForcingDataset)
            else float(
                canonical_floating_array(
                    operand, dtype=self.out_dtype, label=f"{side} expression scalar"
                ).item()
            )
            for side, operand in (("left", self.left), ("right", self.right))
        )
        if (
            self.operation == "div"
            and not isinstance(self._right, ForcingDataset)
            and self._right == 0
        ):
            raise ZeroDivisionError(
                "dataset expression scalar denominator must be nonzero"
            )
        return self

    def _require_compatible(
        self, reference: ForcingDataset, child: ForcingDataset, *, label: str
    ) -> None:
        if child.out_dtype != reference.out_dtype:
            raise ValueError(
                f"{label} has out_dtype {child.out_dtype!r}, expected "
                f"{reference.out_dtype!r}"
            )
        super()._require_compatible(reference, child, label=label)

    def _read_at(self, index: int) -> Any:
        return self._evaluate(index, {}, mode="chunk")

    def _read_source(self, index: int) -> Any:
        return self._evaluate(index, {}, mode="source")

    def _read_item(self, index: int) -> Any:
        return self._evaluate(index, {}, mode="item")

    def _evaluate(
        self,
        index: int,
        reads: dict[int, Any],
        *,
        mode: Literal["chunk", "source", "item"],
    ) -> Any:
        return _evaluate_expression(
            self.operation,
            self._value(self._left, index, reads, mode=mode),
            self._value(self._right, index, reads, mode=mode),
            left_is_scalar=not isinstance(self._left, ForcingDataset),
            right_is_scalar=not isinstance(self._right, ForcingDataset),
            out_dtype=self.out_dtype,
        )

    @staticmethod
    def _value(
        operand: Any,
        index: int,
        reads: dict[int, Any],
        *,
        mode: Literal["chunk", "source", "item"],
    ):
        if not isinstance(operand, ForcingDataset):
            return operand
        if isinstance(operand, DatasetExpression):
            return operand._evaluate(index, reads, mode=mode)
        # Reads are deterministic: an operand used twice is read once.
        if id(operand) not in reads:
            reads[id(operand)] = (
                operand[index]
                if mode == "item"
                else operand._read_source(index)
                if mode == "source"
                else operand._read_at(index)
            )
        return reads[id(operand)]

    def _derived(self, derive: Callable[[ForcingDataset], ForcingDataset]) -> Self:
        views: dict[int, ForcingDataset] = {}

        def mapped(operand: Any) -> Any:
            if not isinstance(operand, ForcingDataset):
                return operand
            if id(operand) not in views:
                views[id(operand)] = derive(operand)
            return views[id(operand)]

        left, right = mapped(self.left), mapped(self.right)
        reference = left if isinstance(left, ForcingDataset) else right
        return self._view(
            {"left": left, "right": right},
            _left=mapped(self._left),
            _right=mapped(self._right),
            _reference=reference,
        )

    def _mapped(self, source_indices: np.ndarray, target_ids: np.ndarray) -> Self:
        return self._derived(lambda child: child._mapped(source_indices, target_ids))

    @validate_call(config=HydroForgeModel.model_config)
    def selected(self, target_ids: UniqueIds) -> Self:
        """Select every point operand together, preserving the expression."""

        if not isinstance(self.space, PointSpace):
            raise TypeError("selected() requires point datasets")
        return self._derived(lambda child: child.selected(target_ids))

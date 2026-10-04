# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Pre-aggregated ``(time, saved_points)`` forcing, e.g. exported rank files."""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import timedelta
from functools import cache, partial
from pathlib import Path
from typing import Annotated, Any, Self

import numpy as np
from pydantic import AfterValidator, Field, PositiveInt, PrivateAttr, validate_call

from hydroforge.core.arrays import UniqueIds, immutable_array
from hydroforge.core.time import DateLike
from hydroforge.core.validation import HydroForgeModel
from hydroforge.data.datasets.base import ForcingDataset, SourceDirectory
from hydroforge.data.datasets.keys import single_file_key
from hydroforge.data.datasets.plan import (
    DatasetPlan,
    SourceChunk,
    TemporalDomain,
    plan_index,
)
from hydroforge.data.datasets.space import PointSpace
from hydroforge.data.datasets.storage import (
    SOURCE_FILE_LABEL,
    NetCDFStore,
    TimeAggregation,
    UnitFactor,
    concatenate_reads,
    scan_storage,
    storage_chunk_len,
)
from hydroforge.data.datasets.timeline import ReadOp, StorageLayout
from hydroforge.io.files import SourceFiles
from hydroforge.io.netcdf.read import (
    configure_variable_cache,
    decoded_element_bytes,
    plan_read_chunk_len,
    prefer_sparse_axis,
    read_variable,
)
from hydroforge.io.rank_output.schema import POINT_DIM, TIME_DIM, read_point_coordinate
from hydroforge.parallel.distributed import is_rank_zero

logger = logging.getLogger(__name__)

_NUMBA_C_THRESHOLD = 5000


def _int64_vector(value: np.ndarray, *, label: str) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        raise ValueError(f"{label} must not be a masked array")
    array = np.asarray(value)
    if array.ndim != 1:
        raise ValueError(f"{label} must be one-dimensional")
    if array.dtype.kind not in {"i", "u"}:
        raise ValueError(f"{label} must contain integers")
    if (
        array.dtype.kind == "u"
        and array.size
        and np.any(array > np.iinfo(np.int64).max)
    ):
        raise ValueError(f"{label} contains a value outside int64 range")
    return immutable_array(array, dtype=np.int64, order="C")


def _window_starts(value: np.ndarray) -> np.ndarray:
    starts = _int64_vector(value, label="window_starts")
    if starts.size == 0:
        raise ValueError("window_starts must contain at least one window")
    if np.any(starts < 0) or np.any(np.diff(starts) <= 0):
        raise ValueError("window_starts must be nonnegative and strictly increasing")
    return starts


_TimeShift = Annotated[
    np.ndarray, AfterValidator(partial(_int64_vector, label="time_shift_steps"))
]


@cache
def _numba_gather() -> Callable[..., np.ndarray]:
    """Compile the shifted-column gather on first use (Numba is optional)."""

    import numba

    @numba.njit(cache=True, parallel=True)
    def gather(data, shift, base_t, length, oob_fill):
        T, C = data.shape
        out = np.empty((length, C), dtype=data.dtype)
        for c in numba.prange(C):
            s = int(shift[c])
            for t in range(length):
                src = base_t + t + s
                out[t, c] = data[src, c] if 0 <= src < T else oob_fill
        return out

    return gather


@dataclass(frozen=True, slots=True)
class _Resident:
    """Main-period and spin-up values in ``out_dtype`` and selection order."""

    main: np.ndarray | dict[str, np.ndarray]
    spinup: np.ndarray | dict[str, np.ndarray] | None


class _PointShards:
    """Validate that every shard stores one variable on one exact point axis."""

    def __init__(self, variable: str, coordinate: str) -> None:
        self._variable = variable
        self._coordinate = coordinate
        self._dtype: np.dtype | None = None
        self.ids: np.ndarray | None = None

    def inspect(self, dataset: Any, path: Path) -> None:
        variable = dataset.variables[self._variable]
        dtype = np.dtype(variable.dtype)
        if dtype.kind not in {"i", "u", "f"}:
            raise ValueError(
                f"Variable {self._variable!r} in {path.name} must use a real "
                f"numeric dtype; got {dtype}"
            )
        if self._dtype is None:
            self._dtype = dtype
        elif dtype != self._dtype:
            raise ValueError(
                f"Variable {self._variable!r} dtype in {path.name} does not "
                f"match the canonical dtype {self._dtype}"
            )
        dimensions = tuple(variable.dimensions)
        if dimensions != (TIME_DIM, POINT_DIM):
            raise ValueError(
                f"Variable {self._variable!r} in {path.name} must have "
                f"dimensions {(TIME_DIM, POINT_DIM)}; got {dimensions}"
            )
        _name, ids = read_point_coordinate(dataset, name=self._coordinate, path=path)
        if self.ids is None:
            self.ids = immutable_array(ids, order="C")
        elif ids.shape != self.ids.shape or not np.array_equal(ids, self.ids):
            raise ValueError(
                f"{self._coordinate} coordinates in shard {path.name} do not "
                "match the canonical shard in content and order"
            )


def _column_window(selection: np.ndarray | None) -> tuple[slice, np.ndarray | None]:
    """The smallest ``saved_points`` slice holding a selection, and its order."""

    if selection is None:
        return slice(None), None
    if selection.size == 0:
        return slice(0, 0), selection
    start = int(selection.min())
    return slice(start, int(selection.max()) + 1), selection - start


class ExportedDataset(ForcingDataset):
    """Pre-aggregated catchment forcing ``(time, saved_points)``.

    Typically reads files written by ``export_catchment_data``:
    ``{prefix}{time_to_key(time)}{suffix}`` holding ``time``, the point
    coordinate ``coord_name`` on ``saved_points`` and ``var_name``.
    :meth:`selected` returns a view in a requested ID order, optionally with a
    per-column time shift; values need no mapping, so :meth:`shard_forcing`
    only checks the batch.  Missing and non-finite values are rejected unless
    ``missing="zero"``.

    Shifted columns and :meth:`windowed` training windows read a resident copy
    (:meth:`load_to_memory`).  The copy stays in its process: DataLoader
    workers started by spawn or forkserver load their own on first use.
    """

    base_dir: SourceDirectory
    var_name: str
    prefix: str
    chunk_len: int | None = Field(default=None, ge=1)
    suffix: str = "rank0.nc"
    time_to_key: Callable[[DateLike], str] = single_file_key
    coord_name: str = "catchment_id"
    in_memory: bool = False
    unit_factor: UnitFactor = 1.0
    time_aggregation: TimeAggregation = None
    window_length: int | None = Field(default=None, ge=1)
    window_starts: Annotated[np.ndarray, AfterValidator(_window_starts)] | None = Field(
        default=None, repr=False
    )

    _store: NetCDFStore = PrivateAttr()
    _space: PointSpace = PrivateAttr()
    _columns: tuple[slice, np.ndarray | None] = PrivateAttr(default=(slice(None), None))
    _shift: np.ndarray | None = PrivateAttr(default=None)
    _shift_groups: list | None = PrivateAttr(default=None)
    _resident: _Resident | None = PrivateAttr(default=None)

    def _compile_plan(self, domain: TemporalDomain) -> DatasetPlan:
        if (self.window_length is None) != (self.window_starts is None):
            raise ValueError(
                "window_length and window_starts must be declared together"
            )
        layout = StorageLayout(
            base_dir=Path(self.base_dir),
            prefix=self.prefix,
            suffix=self.suffix,
            time_to_key=self.time_to_key,
        )
        inspection = SourceFiles.inspect(label=SOURCE_FILE_LABEL)
        points = _PointShards(self.var_name, self.coord_name)
        scan, domain = scan_storage(
            self,
            domain,
            layout,
            inspection,
            storage_offset=timedelta(0),
            inspect_variable=points.inspect,
        )
        plan = self._planned(
            domain,
            storage_chunk_len(
                self.chunk_len,
                domain,
                layout,
                inspection,
                scan,
                storage_offset=timedelta(0),
                plan=partial(
                    plan_read_chunk_len, var_name=self.var_name, profile="points"
                ),
            ),
        )
        self._store = NetCDFStore(
            files=inspection.files(),
            layout=layout,
            timeline=scan.freeze(plan, SourceChunk.source_times),
            unit_factor=self.unit_factor,
            aggregation=self.time_aggregation,
        )
        self._space = PointSpace(ids=points.ids)
        return plan

    def __getstate__(self) -> dict[str, Any]:
        # A resident copy can be gigabytes; each process loads its own.
        state = super().__getstate__()
        state["__pydantic_private__"] = state["__pydantic_private__"] | {
            "_resident": None
        }
        return state

    @property
    def space(self) -> PointSpace:
        return self._space

    # ------------------------------------------------------------------
    # Storage reads
    # ------------------------------------------------------------------
    def _read_source(self, index: int) -> np.ndarray | dict[str, np.ndarray]:
        return self._read_values(self._store.timeline.reads[index], *self._columns)

    def _read_ops(
        self,
        operations: Sequence[ReadOp],
        columns: slice | np.ndarray,
        reorder: np.ndarray | None,
    ) -> np.ndarray:
        """Read ``(time, point)`` values of ``columns`` in selection order."""

        blocks = []
        for key, rows in operations:
            if not rows:
                continue
            with self._store.files.open_netcdf(self._store.path(key)) as dataset:
                variable = dataset.variables[self.var_name]
                file_columns, file_reorder = columns, reorder
                if isinstance(columns, slice) and reorder is not None:
                    positions = columns.start + reorder
                    if prefer_sparse_axis(variable, 1, positions):
                        file_columns, file_reorder = positions, None
                selectors = (np.asarray(rows, dtype=np.int64), file_columns)
                configure_variable_cache(variable, selectors, time_axis=0)
                values = read_variable(variable, selectors)
            blocks.append(values if file_reorder is None else values[:, file_reorder])
        return concatenate_reads(blocks)

    def _read_values(
        self,
        operations: Sequence[ReadOp],
        columns: slice | np.ndarray,
        reorder: np.ndarray | None,
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Read ``operations`` through the value pipeline."""

        rows = sum(len(indices) for _key, indices in operations)
        return self._convert(
            self._ingest(
                self._read_ops(operations, columns, reorder), rows, label="source chunk"
            )
        )

    def _convert(self, values: np.ndarray) -> np.ndarray | dict[str, np.ndarray]:
        return self._store.finish(
            values, out_dtype=self.out_dtype, label="exported dataset"
        )

    def _source_element_bytes(self, operations: Sequence[ReadOp]) -> int:
        """A conservative element width of reading ``operations``."""

        width = np.dtype(self.out_dtype).itemsize
        if self.time_aggregation is not None or self.unit_factor != 1:
            width = max(width, 8)
        for key in dict.fromkeys(key for key, _rows in operations):
            with self._store.files.open_netcdf(self._store.path(key)) as dataset:
                width = max(
                    width, decoded_element_bytes(dataset.variables[self.var_name])
                )
        return width

    def close(self) -> None:
        """Close this process's persistent NetCDF read handles."""

        self._store.files.close()

    # ------------------------------------------------------------------
    # Resident values and windows
    # ------------------------------------------------------------------
    def _as_cache_data(
        self, data: np.ndarray | dict[str, np.ndarray]
    ) -> np.ndarray | dict[str, np.ndarray]:
        if isinstance(data, dict):
            return {
                name: np.ascontiguousarray(block.astype(self.out_dtype, copy=False))
                for name, block in data.items()
            }
        return np.ascontiguousarray(data.astype(self.out_dtype, copy=False))

    @staticmethod
    def _cache_nbytes(cache: np.ndarray | dict[str, np.ndarray]) -> int:
        if isinstance(cache, dict):
            return sum(block.nbytes for block in cache.values())
        return cache.nbytes

    @staticmethod
    def _cache_shape(cache: np.ndarray | dict[str, np.ndarray]) -> Any:
        if isinstance(cache, dict):
            return {name: block.shape for name, block in cache.items()}
        return cache.shape

    def load_to_memory(self) -> None:
        """Keep the main period (and one spin-up cycle) resident in this process.

        The resident values are this view's reads (value policy, aggregation
        and unit conversion applied) in its selection order; a failure
        publishes nothing.
        """

        if self._resident is not None:
            if is_rank_zero():
                logger.info("Exported data is already resident in memory")
            return
        domain = self._plan.domain
        main_ops = self._store.timeline.operations(domain.times())
        main = self._as_cache_data(self._read_values(main_ops, *self._columns))
        # Spin-up can be years away from the main period: keep one compact
        # copy of its interval instead of the gap between them.
        spinup = None
        if domain.spinup is not None:
            spinup = self._as_cache_data(
                self._read_values(
                    self._store.timeline.operations(domain.spinup_times()),
                    *self._columns,
                )
            )
        self._resident = _Resident(main=main, spinup=spinup)
        if is_rank_zero():
            nbytes = self._cache_nbytes(main) + (
                0 if spinup is None else self._cache_nbytes(spinup)
            )
            logger.info(
                "Loaded exported data shape=%s, spin_up_shape=%s from %d "
                "file(s) (%.1f MiB)",
                self._cache_shape(main),
                None if spinup is None else self._cache_shape(spinup),
                len(main_ops),
                nbytes / (1024 * 1024),
            )

    def _read_at(self, index: int) -> np.ndarray | dict[str, np.ndarray]:
        if self._resident is None and (self.in_memory or self._shift is not None):
            self.load_to_memory()
        if self._resident is None:
            return super()._read_at(index)
        chunk = self._plan.chunk_plan.chunks[index]
        return self._distributed(
            self._gather_resident(
                self._resident.spinup
                if chunk.phase == "spinup"
                else self._resident.main,
                chunk.phase_offset,
                chunk.length,
            )
        )

    def read_window(self, base_step: int, length: int) -> np.ndarray | dict:
        """Read ``length`` main-period steps from ``base_step`` (resident, shifted)."""

        if type(base_step) is not int or type(length) is not int:
            raise TypeError("read_window() takes an int base_step and length")
        total = self._plan.domain.count
        if not (0 <= base_step and 0 < length and base_step + length <= total):
            raise ValueError(
                "exported read window must satisfy "
                f"0 <= base_step < base_step + length <= {total}"
            )
        self.load_to_memory()
        return self._distributed(
            self._gather_resident(self._resident.main, base_step, length)
        )

    def __getitem__(self, index: int) -> np.ndarray | dict[str, np.ndarray]:
        """Read chunk ``index``, or window ``index`` of a windowed view."""

        if self.window_starts is None:
            return super().__getitem__(index)
        start = self.window_starts[plan_index(index, len(self), label="window")]
        return self.read_window(int(start), self.window_length)

    def __len__(self) -> int:
        if self.window_starts is not None:
            return int(self.window_starts.size)
        return super().__len__()

    def _gather_resident(
        self,
        resident: np.ndarray | dict[str, np.ndarray],
        base_t: int,
        length: int,
    ) -> np.ndarray | dict[str, np.ndarray]:
        if isinstance(resident, dict):
            return {
                name: self._gather(
                    block, self._shift, base_t, length, groups=self._shift_groups
                )
                for name, block in resident.items()
            }
        return self._gather(
            resident, self._shift, base_t, length, groups=self._shift_groups
        )

    @staticmethod
    def _compile_groups(shift: np.ndarray) -> list:
        """Precompute [(shift_val, col_indices), ...] for fast _gather dispatch."""
        if shift.size == 0:
            return []
        order = np.argsort(shift, kind="stable")
        sorted_shift = shift[order]
        boundaries = np.flatnonzero(sorted_shift[1:] != sorted_shift[:-1]) + 1
        return [
            (int(shift[columns[0]]), columns) for columns in np.split(order, boundaries)
        ]

    @staticmethod
    def _gather(
        data: np.ndarray,
        shift: np.ndarray | None,
        base_t: int,
        length: int,
        oob_fill: float = 0.0,
        *,
        groups: list | None = None,
    ) -> np.ndarray:
        """Gather a ``(length, C)`` window from in-memory ``data``.

        Without ``shift``/``groups``: plain contiguous slice, zero-padded at
        boundaries.

        A shared column shift is one contiguous slice. Small mixed-shift
        selections use their precompiled groups; larger selections use the
        parallel Numba gather without separately prefilling its output.
        """
        T, C = data.shape
        if groups is not None and len(groups) == 1:
            base_t += groups[0][0]
            shift = groups = None
        if shift is None and groups is None:
            lo = max(base_t, 0)
            hi = min(base_t + length, T)
            if lo == base_t and hi == base_t + length:
                return data[lo:hi].copy()
            out = np.full((length, C), oob_fill, dtype=data.dtype)
            if lo < hi:
                out[lo - base_t : hi - base_t] = data[lo:hi]
            return out
        if C >= _NUMBA_C_THRESHOLD:
            return _numba_gather()(data, shift, base_t, length, float(oob_fill))
        out = np.full((length, C), oob_fill, dtype=data.dtype)
        if groups is None:
            groups = ExportedDataset._compile_groups(shift)
        for s, cols in groups:
            src_lo = base_t + s
            clip_lo = max(src_lo, 0)
            clip_hi = min(src_lo + length, T)
            if clip_lo >= clip_hi:
                continue
            out[clip_lo - src_lo : clip_hi - src_lo, cols] = data[clip_lo:clip_hi, cols]
        return out

    # ------------------------------------------------------------------
    # Views
    # ------------------------------------------------------------------
    @validate_call(config=HydroForgeModel.model_config)
    def selected(
        self,
        target_ids: UniqueIds,
        *,
        time_shift_steps: _TimeShift | None = None,
    ) -> Self:
        """Return a view of ``target_ids`` in that order.

        ``time_shift_steps`` gives each selected column an integer source-time
        offset (read from the resident copy).  The view shares this dataset's
        plan and files; a resident copy is shared when the selection is equal.
        Window declarations are not carried over.
        """

        if time_shift_steps is not None and time_shift_steps.shape != target_ids.shape:
            raise ValueError(f"time_shift_steps must have shape {target_ids.shape}")
        space = self._space.select(
            target_ids,
            label=f"exported file {self._store.path(self._store.timeline.keys[0]).name}",
        )
        if is_rank_zero():
            logger.info(
                "Mapped %d catchments from %d in exported file",
                target_ids.size,
                self._space.ids.size,
            )
        shift = (
            None
            if time_shift_steps is None or not time_shift_steps.any()
            else time_shift_steps
        )
        same = self._space.selection is not None and np.array_equal(
            self._space.selection, space.selection
        )
        return self._view(
            {"window_length": None, "window_starts": None},
            _space=space,
            _target_ids=immutable_array(target_ids),
            _columns=_column_window(space.selection),
            _shift=shift,
            _shift_groups=None if shift is None else self._compile_groups(shift),
            _resident=self._resident if same else None,
        )

    @validate_call(config=HydroForgeModel.model_config)
    def windowed(self, window: PositiveInt, stride: PositiveInt | None = None) -> Self:
        """Return a view whose items are shifted ``window``-step training windows.

        Window ``index`` covers ``[starts[index], starts[index] + window)`` on
        the main axis with ``starts = arange(0, T - window + 1, stride)``;
        shuffle them with the DataLoader.  The values are made resident first,
        so fork-started workers share them copy-on-write.
        """

        total = self._plan.domain.count
        if window > total:
            raise ValueError(f"window={window} exceeds total time steps {total}")
        stride = window if stride is None else stride
        starts = np.arange(0, total - window + 1, stride, dtype=np.int64)
        if is_rank_zero():
            logger.info(
                "Enabled window sampling: window=%d, stride=%d, windows=%d, "
                "time_steps=%d",
                window,
                stride,
                starts.size,
                total,
            )
        self.load_to_memory()
        self.close()
        return self._view(
            {"window_length": window, "window_starts": immutable_array(starts)}
        )

    def filtered(self, keep: np.ndarray) -> Self:
        """Return a view retaining the windows where ``keep`` is true."""

        if self.window_starts is None:
            raise ValueError("filtered() requires a windowed Dataset")
        if (
            np.ma.isMaskedArray(keep)
            or not isinstance(keep, np.ndarray)
            or keep.dtype != np.dtype(np.bool_)
            or keep.shape != self.window_starts.shape
        ):
            raise ValueError(
                "window filter must be a boolean array with shape "
                f"{self.window_starts.shape}"
            )
        return self._view({"window_starts": immutable_array(self.window_starts[keep])})

# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Gridded forcing stored as ``(time, lat, lon)`` NetCDF shards."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import Field, PrivateAttr

from hydroforge.core.arrays import canonical_float64, immutable_array
from hydroforge.core.time import DateLike
from hydroforge.data.datasets.base import ForcingDataset, SourceDirectory
from hydroforge.data.datasets.keys import yearly_time_to_key
from hydroforge.data.datasets.plan import DatasetPlan, SourceChunk, TemporalDomain
from hydroforge.data.datasets.space import GridSpace
from hydroforge.data.datasets.storage import (
    SOURCE_FILE_LABEL,
    NetCDFStore,
    TimeAggregation,
    UnitFactor,
    concatenate_reads,
    scan_storage,
    storage_chunk_len,
)
from hydroforge.data.datasets.timeline import ReadOp, StorageLayout, TimelineScan
from hydroforge.data.datasets.values import MissingPolicy
from hydroforge.io.files import SourceFiles
from hydroforge.io.netcdf.coordinates import LATITUDE_NAMES, LONGITUDE_NAMES, read_axis
from hydroforge.io.netcdf.read import (
    configure_variable_cache,
    plan_read_chunk_len,
    read_variable,
)


@dataclass(frozen=True, slots=True)
class _Shard:
    """Axis positions ``(t, y, x)`` and read-tile shape of one file's variable."""

    axes: tuple[int, int, int]
    tile: tuple[int, int, bool]


@dataclass(frozen=True, slots=True)
class _SelectionRead:
    """How a selection reads its bounding box, possibly as sparse tiles."""

    y: slice
    x: slice
    positions: np.ndarray
    tiles: Mapping[tuple[int, int, bool], tuple | None]


def _pick_dimension(dimensions: tuple[str, ...], *names: str) -> int | None:
    matches = [(name, dimensions.index(name)) for name in names if name in dimensions]
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous dimensions {[name for name, _ in matches]} in {dimensions}"
        )
    return None if not matches else matches[0][1]


class _GridShards:
    """Validate that every shard stores one variable on one exact grid."""

    def __init__(self, variable: str) -> None:
        self._variable = variable
        self._dtype: np.dtype | None = None
        self._grid: tuple[Any, ...] | None = None
        self.shards: dict[Path, _Shard] = {}

    def inspect(self, dataset: Any, path: Path) -> None:
        variable = dataset.variables[self._variable]
        dtype = np.dtype(variable.dtype)
        if dtype.kind not in {"i", "u", "f"}:
            raise ValueError(
                f"NetCDF forcing variable {self._variable!r} in {path.name} "
                f"must use a real numeric dtype; got {dtype}"
            )
        if self._dtype is None:
            self._dtype = dtype
        elif dtype != self._dtype:
            raise ValueError(
                f"NetCDF forcing dtype in shard {path.name} does not match "
                f"the canonical dtype {self._dtype}"
            )
        dimensions = tuple(variable.dimensions)
        axes = (
            _pick_dimension(dimensions, "time", "valid_time"),
            _pick_dimension(dimensions, *LATITUDE_NAMES),
            _pick_dimension(dimensions, *LONGITUDE_NAMES),
        )
        if len(dimensions) != 3 or None in axes:
            raise ValueError(
                "NetCDF forcing variable must have exactly one time, one "
                f"latitude, and one longitude dimension; got {dimensions}"
            )
        _time, y_axis, x_axis = axes
        coordinates = []
        for label, names, axis in (
            ("longitude", LONGITUDE_NAMES, x_axis),
            ("latitude", LATITUDE_NAMES, y_axis),
        ):
            coordinate = read_axis(
                dataset, names, dimension=dimensions[axis], path=path, label=label
            )
            values = canonical_float64(
                coordinate.values, label=f"{label} coordinate in {path.name}"
            )
            if np.unique(values).size != values.size:
                raise ValueError(
                    f"{label} coordinate in {path.name} contains duplicate values"
                )
            bounds = coordinate.bounds
            if bounds is not None:
                bounds = canonical_float64(
                    bounds, label=f"{label} bounds in {path.name}"
                )
            coordinates.append((values, coordinate.units, bounds))
        grid = (
            tuple(values for values, _units, _bounds in coordinates),
            tuple(units for _values, units, _bounds in coordinates),
            tuple(bounds for _values, _units, bounds in coordinates),
            (variable.shape[y_axis], variable.shape[x_axis]),
        )
        if self._grid is None:
            self._grid = grid
        else:
            self._require_same_grid(grid, path)
        chunking = variable.chunking()
        chunked = chunking != "contiguous" and bool(chunking)
        self.shards[path] = _Shard(
            axes=axes,
            tile=(
                int(chunking[y_axis]) if chunked else 1,
                int(chunking[x_axis]) if chunked else variable.shape[x_axis],
                chunked,
            ),
        )

    def _require_same_grid(self, grid: tuple[Any, ...], path: Path) -> None:
        axes, units, bounds, shape = grid
        expected_axes, expected_units, expected_bounds, expected_shape = self._grid
        if not all(
            left.shape == right.shape and np.array_equal(left, right)
            for left, right in zip(expected_axes, axes, strict=True)
        ):
            raise ValueError(
                f"spatial coordinates in shard {path.name} do not match "
                "the canonical shard in content and order"
            )
        if units != expected_units:
            raise ValueError(
                f"spatial coordinate units in shard {path.name} do not "
                "match the canonical shard"
            )
        if not all(
            (left is None and right is None)
            or (
                left is not None
                and right is not None
                and left.shape == right.shape
                and np.array_equal(left, right)
            )
            for left, right in zip(expected_bounds, bounds, strict=True)
        ):
            raise ValueError(
                f"spatial coordinate bounds in shard {path.name} do not "
                "match the canonical shard"
            )
        if shape != expected_shape:
            raise ValueError(
                f"spatial shape in shard {path.name} does not match the canonical shard"
            )

    def space(self) -> GridSpace:
        (longitude, latitude), units, (lon_bounds, lat_bounds), _shape = self._grid
        return GridSpace(
            longitude_units=units[0],
            latitude_units=units[1],
            longitude=immutable_array(longitude, order="C"),
            latitude=immutable_array(latitude, order="C"),
            longitude_bounds=(
                None if lon_bounds is None else immutable_array(lon_bounds, order="C")
            ),
            latitude_bounds=(
                None if lat_bounds is None else immutable_array(lat_bounds, order="C")
            ),
        )


def _tile_plan(
    rows: np.ndarray,
    columns: np.ndarray,
    bbox: tuple[int, int, int, int],
    tile: tuple[int, int, bool],
    grid_width: int,
) -> tuple | None:
    """Read tiles when they cover much less than the selection bounding box."""

    tile_height, tile_width, chunked = tile
    minimum_y, maximum_y, minimum_x, maximum_x = bbox
    bbox_area = (maximum_y - minimum_y + 1) * (maximum_x - minimum_x + 1)
    if rows.size * 4 >= bbox_area:
        return None
    tile_ids = (rows // tile_height) * (
        (grid_width + tile_width - 1) // tile_width
    ) + columns // tile_width
    order = np.argsort(tile_ids, kind="stable")
    boundaries = np.flatnonzero(np.diff(tile_ids[order]) != 0) + 1
    if boundaries.size >= 128:
        return None
    tiles = []
    covered = 0
    for positions in np.split(order, boundaries):
        start_y, stop_y = int(rows[positions].min()), int(rows[positions].max()) + 1
        start_x = int(columns[positions].min())
        stop_x = int(columns[positions].max()) + 1
        local = (rows[positions] - start_y) * (stop_x - start_x) + (
            columns[positions] - start_x
        )
        tiles.append((slice(start_y, stop_y), slice(start_x, stop_x), positions, local))
        covered += (stop_y - start_y) * (stop_x - start_x)
    physical_bbox = (maximum_y // tile_height - minimum_y // tile_height + 1) * (
        maximum_x // tile_width - minimum_x // tile_width + 1
    )
    if covered * 2 < bbox_area and (not chunked or len(tiles) * 2 < physical_bbox):
        return tuple(tiles)
    return None


def _selection_read(
    space: GridSpace, tiles: set[tuple[int, int, bool]]
) -> _SelectionRead | None:
    """Compile the bounding box and tile reads of one non-empty selection."""

    selection = space.selection
    if selection is None or selection.size == 0:
        return None
    grid_width = space.shape[1]
    rows, columns = np.divmod(selection, grid_width)
    bbox = (int(rows.min()), int(rows.max()), int(columns.min()), int(columns.max()))
    minimum_y, maximum_y, minimum_x, maximum_x = bbox
    positions = (rows - minimum_y) * (maximum_x - minimum_x + 1) + (columns - minimum_x)
    return _SelectionRead(
        y=slice(minimum_y, maximum_y + 1),
        x=slice(minimum_x, maximum_x + 1),
        positions=immutable_array(positions, dtype=np.int64),
        tiles={
            tile: _tile_plan(rows, columns, bbox, tile, grid_width) for tile in tiles
        },
    )


class NetCDFDataset(ForcingDataset):
    """Gridded ``(time, lat, lon)`` NetCDF forcing, one variable per dataset.

    Only the time variables of the files the plan needs are scanned at
    construction; each chunk's reads are compiled once and group consecutive
    times of one file into a single read.  A mapped view reads only the
    bounding box of its source cells, or sparse tiles of it.  Missing values
    become zero unless ``missing="error"``.
    """

    base_dir: SourceDirectory
    var_name: str
    prefix: str
    chunk_len: int | None = Field(default=None, ge=1)
    unit_factor: UnitFactor = 1.0
    suffix: str = ".nc"
    time_to_key: Callable[[DateLike], str] = yearly_time_to_key
    time_aggregation: TimeAggregation = None
    missing: MissingPolicy = "zero"

    _store: NetCDFStore = PrivateAttr()
    _shards: Mapping[Path, _Shard] = PrivateAttr()
    _space: GridSpace = PrivateAttr()
    _selection_read: _SelectionRead | None = PrivateAttr(default=None)

    def _compile_plan(self, domain: TemporalDomain) -> DatasetPlan:
        layout = StorageLayout(
            base_dir=Path(self.base_dir),
            prefix=self.prefix,
            suffix=self.suffix,
            time_to_key=self.time_to_key,
        )
        inspection = SourceFiles.inspect(label=SOURCE_FILE_LABEL)
        grid = _GridShards(self.var_name)
        offset = self._storage_offset()
        scan, domain = scan_storage(
            self,
            domain,
            layout,
            inspection,
            storage_offset=offset,
            inspect_variable=grid.inspect,
        )
        plan = self._planned(
            domain,
            storage_chunk_len(
                self.chunk_len,
                domain,
                layout,
                inspection,
                scan,
                storage_offset=offset,
                plan=self._planned_chunk_len,
            ),
        )
        self._locate_support(scan, plan)
        self._store = NetCDFStore(
            files=inspection.files(),
            layout=layout,
            timeline=scan.freeze(plan, self._read_times),
            unit_factor=self.unit_factor,
            aggregation=self.time_aggregation,
        )
        self._shards = grid.shards
        self._space = grid.space()
        return plan

    def _storage_offset(self) -> timedelta:
        """Storage time minus logical time of each record."""

        return timedelta(0)

    def _planned_chunk_len(self, path: Path) -> int:
        """Automatic source records per read for this storage kind."""

        return plan_read_chunk_len(path, self.var_name)

    def _read_times(self, chunk: SourceChunk) -> Sequence[DateLike]:
        """Storage times that one chunk reads."""

        return chunk.source_times()

    def _locate_support(self, scan: TimelineScan, plan: DatasetPlan) -> None:
        """Locate storage times read beyond the plan's own samples."""

    @property
    def space(self) -> GridSpace:
        return self._space

    def _mapped(self, source_indices: np.ndarray, target_ids: np.ndarray):
        space = self._space.select(source_indices)
        return self._view(
            _space=space,
            _target_ids=target_ids,
            _selection_read=_selection_read(
                space, {shard.tile for shard in self._shards.values()}
            ),
        )

    def _read_source(self, index: int) -> np.ndarray | dict[str, np.ndarray]:
        # An aggregated chunk reads ``aggregation_factor`` records per row.
        operations = self._store.timeline.reads[index]
        return self._converted(
            self._read_operations(operations),
            sum(len(rows) for _key, rows in operations),
        )

    def _read_operations(self, operations: Sequence[ReadOp]) -> np.ndarray:
        """Read ``(T, Y, X)``, or ``(T, N)`` for a selection, without conversion."""

        selection = self._space.selection
        if selection is not None and selection.size == 0:
            rows = sum(len(indices) for _key, indices in operations)
            return np.empty((rows, 0), dtype=self.out_dtype)
        return concatenate_reads(
            [self._read_file(key, rows) for key, rows in operations if rows]
        )

    def _read_file(self, key: str, rows: tuple[int, ...]) -> np.ndarray:
        path = self._store.path(key)
        shard = self._shards[path]
        axes = shard.axes
        time_axis, y_axis, x_axis = axes
        read = self._selection_read
        with self._store.files.open_netcdf(path) as dataset:
            variable = dataset.variables[self.var_name]
            selectors: list[Any] = [slice(None)] * 3
            selectors[time_axis] = np.asarray(rows, dtype=np.int64)
            if read is None:
                configure_variable_cache(
                    variable, tuple(selectors), time_axis=time_axis
                )
                return np.transpose(read_variable(variable, tuple(selectors)), axes)
            selectors[y_axis], selectors[x_axis] = read.y, read.x
            tiles = read.tiles[shard.tile]
            if tiles is not None:
                return self._read_tiles(variable, selectors, axes, tiles)
            configure_variable_cache(variable, tuple(selectors), time_axis=time_axis)
            values = np.transpose(read_variable(variable, tuple(selectors)), axes)
            return values.reshape(values.shape[0], -1)[:, read.positions]

    def _read_tiles(
        self,
        variable: Any,
        selectors: list[Any],
        axes: tuple[int, int, int],
        tiles: tuple,
    ) -> np.ndarray:
        time_axis, y_axis, x_axis = axes
        values = mask = None
        for y_slice, x_slice, positions, local in tiles:
            tile_selectors = list(selectors)
            tile_selectors[y_axis], tile_selectors[x_axis] = y_slice, x_slice
            configure_variable_cache(
                variable, tuple(tile_selectors), time_axis=time_axis
            )
            tile = np.transpose(read_variable(variable, tuple(tile_selectors)), axes)
            block = tile.reshape(tile.shape[0], -1)[:, local]
            if values is None:
                values = np.empty((block.shape[0], self._space.size), dtype=block.dtype)
            values[:, positions] = np.ma.getdata(block)
            if np.ma.is_masked(block):
                if mask is None:
                    mask = np.zeros(values.shape, dtype=bool)
                mask[:, positions] = np.ma.getmaskarray(block)
        return values if mask is None else np.ma.MaskedArray(values, mask)

    def _convert(self, values: np.ndarray) -> np.ndarray | dict[str, np.ndarray]:
        return self._store.finish(
            values, out_dtype=self.out_dtype, label="NetCDF dataset"
        )

    def _first_frame_missing(self) -> np.ndarray:
        """``(Y, X)`` mask of missing values in the first read source frame."""

        first = self.chunk_plan[0].source_start + self._storage_offset()
        key, rows = self._store.timeline.operations([first])[0]
        path = self._store.path(key)
        axes = self._shards[path].axes
        with self._store.files.open_netcdf(path) as dataset:
            variable = dataset.variables[self.var_name]
            selectors: list[Any] = [slice(None)] * 3
            selectors[axes[0]] = np.asarray(rows[:1], dtype=np.int64)
            configure_variable_cache(variable, tuple(selectors), time_axis=axes[0])
            frame = np.transpose(read_variable(variable, tuple(selectors)), axes)[0]
        if np.ma.isMaskedArray(frame):
            return np.ma.getmaskarray(frame) | np.isnan(np.ma.getdata(frame))
        if frame.dtype.kind != "f":
            return np.zeros(frame.shape, dtype=bool)
        return np.isnan(frame)

    def close(self) -> None:
        """Close this process's persistent NetCDF read handles."""

        self._store.files.close()

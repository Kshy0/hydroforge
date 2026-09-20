"""Rank-file discovery, timeline validation, and coordinate resolution."""

from __future__ import annotations

import logging
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import netCDF4 as nc
import numpy as np
from pydantic import BeforeValidator, Field, field_validator, model_validator

from hydroforge.contracts.temporal import canonical_calendar
from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.serialization.netcdf import (
    BOOL_LOGICAL_DTYPE,
    COMMITTED_STEPS_ATTR,
    LOGICAL_DTYPE_ATTR,
    OUTPUT_FORMAT,
    OUTPUT_VERSION,
    RUN_ID_ATTR,
)

logger = logging.getLogger(__name__)


def _native_contract_int(value: Any) -> int:
    if type(value) not in {int, np.int32, np.int64}:
        raise ValueError("must be a Python, int32 or int64 integer")
    return int(value)


def _native_committed_int(value: Any) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise ValueError(f"{COMMITTED_STEPS_ATTR} must be an integer")
    return int(value)


_NativeContractInt = Annotated[int, BeforeValidator(_native_contract_int)]


class _RankFileDeclaration(HydroForgeModel):
    output_format: str = Field(validation_alias="hydroforge_output_format")
    output_version: _NativeContractInt = Field(
        validation_alias="hydroforge_output_version"
    )
    rank: _NativeContractInt = Field(ge=0, validation_alias="hydroforge_rank")
    world_size: _NativeContractInt = Field(
        ge=1, validation_alias="hydroforge_world_size"
    )
    run_id: str = Field(min_length=1, validation_alias=RUN_ID_ATTR)
    committed_steps: Annotated[int, BeforeValidator(_native_committed_int)] = Field(
        ge=0, validation_alias=COMMITTED_STEPS_ATTR
    )

    @field_validator("run_id")
    @classmethod
    def _validate_run_id(cls, value: str) -> str:
        if value != value.strip():
            raise ValueError(f"{RUN_ID_ATTR} must not contain surrounding whitespace")
        return value

    @model_validator(mode="after")
    def _validate_contract(self):
        if self.output_format != OUTPUT_FORMAT:
            raise ValueError(f"unsupported output format {self.output_format!r}")
        if self.output_version != OUTPUT_VERSION:
            raise ValueError(
                f"unsupported output version {self.output_version}; version "
                f"{OUTPUT_VERSION} with a shared run identity is required"
            )
        if self.rank >= self.world_size:
            raise ValueError("invalid hydroforge_world_size/rank contract")
        return self


class RankOutputCatalog:
    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def _select_coord_name(self, ds: nc.Dataset, saved_points: int) -> str | None:
        """Pick a ('saved_points',) variable to serve as output_coord."""
        if self.owner.coord_name:
            variable = ds.variables[self.owner.coord_name]
            if (
                variable.dimensions != ("saved_points",)
                or len(variable) != saved_points
            ):
                raise ValueError(
                    f"requested coordinate {self.owner.coord_name!r} must have "
                    "dimensions ('saved_points',)"
                )
            return self.owner.coord_name

        candidates: list[str] = []
        for name, v in ds.variables.items():
            if name in ("time", self.owner.var_name):
                continue
            if v.dimensions == ("saved_points",) and len(v) == saved_points:
                value = v[:]
                if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
                    if np.dtype(v.dtype).kind in "iu":
                        raise ValueError(
                            f"coordinate candidate {name!r} contains missing values"
                        )
                    continue
                array = np.asarray(value)
                if array.dtype.kind in "iu" and np.unique(array).size == array.size:
                    candidates.append(name)
        if len(candidates) > 1:
            raise ValueError(
                "multiple integer saved_points coordinates are eligible for "
                f"automatic selection: {candidates}; specify coord_name"
            )
        return candidates[0] if candidates else None

    def _inspect_rank_file(
        self,
        ds: nc.Dataset,
        *,
        expected: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        declaration = _RankFileDeclaration.model_validate(
            {
                field.validation_alias: ds.getncattr(field.validation_alias)
                for field in _RankFileDeclaration.model_fields.values()
            }
        )
        variable = ds.variables[self.owner.var_name]
        logical_dtype = getattr(variable, LOGICAL_DTYPE_ATTR, None)
        if logical_dtype is not None:
            if logical_dtype != BOOL_LOGICAL_DTYPE:
                raise ValueError(
                    f"variable {self.owner.var_name!r} declares "
                    f"unsupported logical dtype {logical_dtype!r}"
                )
        committed = declaration.committed_steps
        time_length = len(ds.variables["time"])
        data_length = len(variable)
        if committed > time_length or data_length != time_length:
            raise ValueError(
                "rank output has an uncommitted or inconsistent append: "
                f"committed={committed}, time={time_length}, "
                f"data={data_length}"
            )
        # The optional trailing value axis keeps its model-defined name.
        dimensions = tuple(variable.dimensions)
        cursor = 0
        if not dimensions or dimensions[cursor] != "time":
            raise ValueError(
                f"variable {self.owner.var_name!r} has dimensions "
                f"{dimensions}; the first dimension must be 'time'"
            )
        cursor += 1
        has_ensemble = cursor < len(dimensions) and dimensions[cursor] == "ensemble"
        if has_ensemble:
            cursor += 1
        if cursor >= len(dimensions) or dimensions[cursor] != "saved_points":
            expected_layout = "('time', ['ensemble'], 'saved_points', [value_axis])"
            raise ValueError(
                f"variable {self.owner.var_name!r} has dimensions "
                f"{dimensions}, expected {expected_layout}"
            )
        cursor += 1
        trailing_dimensions = dimensions[cursor:]
        if len(trailing_dimensions) > 1:
            raise ValueError(
                f"variable {self.owner.var_name!r} has multiple trailing "
                f"value dimensions {trailing_dimensions}; the reader's "
                "level API supports at most one"
            )
        level_dimension = trailing_dimensions[0] if trailing_dimensions else None
        has_levels = level_dimension is not None
        saved_points = int(ds.dimensions["saved_points"].size)
        metadata = {
            "saved_points": saved_points,
            "has_ensemble": has_ensemble,
            "member_count": (
                int(ds.dimensions["ensemble"].size) if has_ensemble else 0
            ),
            "has_levels": has_levels,
            "n_levels": (
                int(ds.dimensions[level_dimension].size)
                if level_dimension is not None
                else 0
            ),
            "level_dimension": level_dimension,
            "dimensions": dimensions,
            "dtype": np.dtype(variable.dtype),
            "logical_dtype": logical_dtype,
            "contract_rank": declaration.rank,
            "world_size": declaration.world_size,
            "run_id": declaration.run_id,
            "committed_steps": committed,
        }
        if expected is not None:
            schema = dict(metadata)
            schema.pop("committed_steps")
            if schema != expected:
                raise ValueError(
                    f"rank output schema differs from {expected}: {schema}"
                )
        return metadata

    @staticmethod
    def _read_coordinate(
        dataset: nc.Dataset,
        name: str,
        saved_points: int,
    ) -> np.ndarray:
        variable = dataset.variables[name]
        if variable.dimensions != ("saved_points",) or len(variable) != saved_points:
            raise ValueError(
                f"coordinate {name!r} must have dimensions ('saved_points',)"
            )
        raw = variable[:]
        if np.ma.isMaskedArray(raw) and np.any(np.ma.getmaskarray(raw)):
            raise ValueError(f"coordinate {name!r} contains missing values")
        coordinate = RankOutputCatalog._coordinate_int64(
            np.asarray(raw),
            label=f"coordinate {name!r}",
            expected_shape=(saved_points,),
        )
        if np.unique(coordinate).size != coordinate.size:
            raise ValueError(f"coordinate {name!r} contains duplicate IDs")
        return coordinate

    def scan(self, paths: Sequence[Path]) -> list[dict]:
        """Validate the captured file set and collect structural metadata."""
        rank_map: dict[int, dict[int, Path]] = {}

        # Regex to match rank and optional year: var_rank0.nc or var_rank0_2000.nc
        if self.owner.split_by_year:
            rank_re = re.compile(
                rf"^{re.escape(self.owner.var_name)}_rank(\d+)_(-?\d+)\.nc$"
            )
        else:
            rank_re = re.compile(rf"^{re.escape(self.owner.var_name)}_rank(\d+)\.nc$")

        for fp in paths:
            m = rank_re.match(fp.name)
            if not m:
                raise ValueError(
                    f"candidate output file {fp.name!r} does not match the "
                    f"configured {'year-split' if self.owner.split_by_year else 'single-file'} "
                    "rank naming contract"
                )
            rank_id = int(m.group(1))
            if self.owner.split_by_year:
                year = int(m.group(2))
            else:
                year = -1

            rank_paths = rank_map.setdefault(rank_id, {})
            if year in rank_paths:
                raise ValueError(
                    f"duplicate output files for rank {rank_id}"
                    + (f" and year {year}" if self.owner.split_by_year else "")
                )
            rank_paths[year] = fp

        if rank_map:
            observed = sorted(rank_map)
            if any(rank != position for position, rank in enumerate(observed)):
                raise ValueError(
                    "rank output files must form a contiguous set starting at "
                    f"zero: expected {len(observed)} consecutive ranks, found {observed}"
                )

        rank_infos: list[dict] = []

        for rank_id in sorted(rank_map.keys()):
            # Sort files by year (or just by name if year is -1, but here we use the tuple)
            # If year is -1, it means no year suffix.
            files_with_year = sorted(rank_map[rank_id].items())
            years = tuple(year for year, _path in files_with_year)
            paths = [path for _year, path in files_with_year]

            # Use the first file to get metadata
            first_fp = paths[0]

            try:
                metadata = None
                committed_steps = []
                coord_name = coord_raw = None
                for path in paths:
                    with nc.Dataset(path, "r") as dataset:
                        observed = self._inspect_rank_file(dataset, expected=metadata)
                        committed_steps.append(observed.pop("committed_steps"))
                        first_file = metadata is None
                        if first_file:
                            metadata = observed
                            if metadata["contract_rank"] != rank_id:
                                raise ValueError(
                                    f"file name rank {rank_id} disagrees with contract rank "
                                    f"{metadata['contract_rank']}"
                                )
                        saved_points = metadata["saved_points"]
                        observed_name = self._select_coord_name(dataset, saved_points)
                        observed_coordinate = (
                            None
                            if observed_name is None
                            else self._read_coordinate(
                                dataset, observed_name, saved_points
                            )
                        )
                        if first_file:
                            coord_name, coord_raw = observed_name, observed_coordinate
                        elif observed_name != coord_name:
                            raise ValueError(
                                f"coordinate name changes from {coord_name!r} "
                                f"to {observed_name!r} in {path.name}"
                            )
                        elif coord_name is not None and not np.array_equal(
                            observed_coordinate, coord_raw
                        ):
                            raise ValueError(
                                f"coordinate {coord_name!r} changes values or order in {path.name}"
                            )

                rank_infos.append(
                    {
                        "rank_id": rank_id,
                        "years": years,
                        "paths": paths,
                        "file_committed_steps": tuple(committed_steps),
                        **metadata,
                        "coord_name": coord_name,
                        "coord_raw": coord_raw,
                        "x": None,
                        "y": None,
                    }
                )
            except (
                OSError,
                KeyError,
                OverflowError,
                TypeError,
                ValueError,
            ) as exc:
                raise ValueError(
                    f"Failed to inspect rank {rank_id} file {first_fp}"
                ) from exc

        if rank_infos:
            reference = rank_infos[0]
            for info in rank_infos[1:]:
                for name in (
                    "years",
                    "has_ensemble",
                    "member_count",
                    "has_levels",
                    "n_levels",
                    "level_dimension",
                    "dimensions",
                    "dtype",
                    "logical_dtype",
                    "coord_name",
                    "world_size",
                    "run_id",
                ):
                    if info[name] != reference[name]:
                        raise ValueError(
                            f"rank {info['rank_id']} output {name} differs from "
                            f"rank 0: {info[name]!r} != {reference[name]!r}"
                        )
            world_size = reference["world_size"]
            observed_ranks = [info["rank_id"] for info in rank_infos]
            if len(observed_ranks) != world_size:
                raise ValueError(
                    "rank output set is incomplete for declared world_size: "
                    f"expected {world_size} ranks, found {len(observed_ranks)}: "
                    f"{observed_ranks}"
                )
            coordinates = [
                info["coord_raw"]
                for info in rank_infos
                if info["coord_raw"] is not None
            ]
            if coordinates:
                combined = np.concatenate(coordinates)
                if np.unique(combined).size != combined.size:
                    raise ValueError(
                        "output coordinates contain duplicate IDs across rank files"
                    )

        return rank_infos

    def read_timeline(self) -> None:
        """
        Read the time axis and require exact agreement across every rank.
        Produce:
          - self.owner._time_values_num
          - self.owner._time_datetimes (naive)
          - self.owner._time_units / _time_calendar
          - self.owner._time_len
        """
        rank_timelines = []
        for info in self.owner._rank_files:
            datetimes = []
            offsets = []
            current_offset = 0
            calendar = None
            first_units = None
            for path, declared_year, committed_steps in zip(
                info["paths"],
                info["years"],
                info["file_committed_steps"],
                strict=True,
            ):
                with nc.Dataset(path, "r") as dataset:
                    variable = dataset.variables["time"]
                    units = getattr(variable, "units", None)
                    if not isinstance(units, str) or not units.strip():
                        raise ValueError(
                            f"time variable in {path.name} has no CF units"
                        )
                    if first_units is None:
                        first_units = units
                    file_calendar = canonical_calendar(
                        getattr(variable, "calendar", "standard"),
                    )
                    if calendar is None:
                        calendar = file_calendar
                    elif file_calendar != calendar:
                        raise ValueError(
                            f"rank {info['rank_id']} files use inconsistent calendars"
                        )
                    raw_values = variable[:committed_steps]
                    if np.ma.isMaskedArray(raw_values) and np.any(
                        np.ma.getmaskarray(raw_values)
                    ):
                        raise ValueError(
                            f"time variable in {path.name} contains missing values"
                        )
                    values = np.asarray(raw_values)
                    if values.ndim != 1:
                        raise ValueError(f"time variable in {path.name} must be 1-D")
                    if (
                        values.dtype.kind not in "iuf"
                        or not np.isfinite(
                            values,
                        ).all()
                    ):
                        raise ValueError(
                            f"time variable in {path.name} must contain finite "
                            "numeric values"
                        )
                    decoded = list(
                        nc.num2date(
                            values,
                            units=units,
                            calendar=file_calendar,
                        )
                    )
                    if self.owner.split_by_year and any(
                        instant.year != declared_year for instant in decoded
                    ):
                        observed_years = sorted({instant.year for instant in decoded})
                        raise ValueError(
                            f"year-split file {path.name} declares year "
                            f"{declared_year} but contains timestamps from "
                            f"{observed_years}"
                        )
                    datetimes.extend(decoded)
                    length = len(decoded)
                    offsets.append((current_offset, current_offset + length))
                    current_offset += length
            if any(right <= left for left, right in zip(datetimes, datetimes[1:])):
                raise ValueError(
                    f"rank {info['rank_id']} output time axis must be strictly "
                    "increasing across files"
                )
            info["file_time_offsets"] = tuple(offsets)
            rank_timelines.append((info, datetimes, first_units, calendar))

        common_length = min(
            len(datetimes) for _info, datetimes, _units, _calendar in rank_timelines
        )
        if common_length == 0:
            raise ValueError("rank outputs have no common committed time steps")
        _info, reference_datetimes, time_units, time_calendar = rank_timelines[0]
        master_datetimes = reference_datetimes[:common_length]
        master_values = np.asarray(
            nc.date2num(
                master_datetimes,
                time_units,
                time_calendar,
            )
        )
        for info, datetimes, _units, calendar in rank_timelines[1:]:
            if calendar != time_calendar:
                raise ValueError(
                    f"rank {info['rank_id']} output calendar {calendar!r} "
                    f"differs from rank 0 {time_calendar!r}"
                )
            observed = np.asarray(
                nc.date2num(
                    datetimes[:common_length],
                    time_units,
                    time_calendar,
                )
            )
            if not np.array_equal(observed, master_values):
                raise ValueError(
                    f"rank {info['rank_id']} output timestamps differ from rank 0 "
                    "within the common committed prefix"
                )
        self.owner._time_units = time_units
        self.owner._time_calendar = time_calendar
        self.owner._time_datetimes = master_datetimes
        self.owner._time_values_num = master_values
        self.owner._time_len = common_length

    @staticmethod
    def _coordinate_int64(
        value: Any,
        *,
        label: str,
        expected_shape: tuple[int, ...],
    ) -> np.ndarray:
        if np.ma.isMaskedArray(value):
            raise TypeError(f"{label} must not be a masked array")
        array = np.asarray(value)
        if array.shape != expected_shape:
            raise ValueError(
                f"{label} must have shape {expected_shape}; got {array.shape}"
            )
        if array.dtype.kind not in "iu":
            raise TypeError(f"{label} must contain integers")
        if array.dtype.kind == "u" and np.any(array > np.iinfo(np.int64).max):
            raise OverflowError(f"{label} contains values outside int64 range")
        return np.array(array, dtype=np.int64, order="C", copy=True)

    def compute_coordinates(self) -> None:
        """Compute (x, y) for each rank (custom converter -> unravel -> None)."""
        for info in self.owner._rank_files:
            if info["coord_raw"] is None or info["saved_points"] == 0:
                info["x"], info["y"] = None, None
                continue
            if self.owner.coord_converter is not None:
                x, y = self.owner.coord_converter(info["coord_raw"].copy())
                expected = (info["saved_points"],)
                info["x"] = self._coordinate_int64(
                    x,
                    label=(f"coordinate converter x output for rank {info['rank_id']}"),
                    expected_shape=expected,
                )
                info["y"] = self._coordinate_int64(
                    y,
                    label=(f"coordinate converter y output for rank {info['rank_id']}"),
                    expected_shape=expected,
                )
                continue

            if self.owner.map_shape is not None:
                nx_, ny_ = self.owner.map_shape
                total = nx_ * ny_
                flat = info["coord_raw"]
                valid = (flat >= 0) & (flat < total)
                if not np.all(valid):
                    invalid = int(flat[np.flatnonzero(~valid)[0]])
                    raise ValueError(
                        f"linear coordinate {invalid} for rank "
                        f"{info['rank_id']} is outside map_shape index range "
                        f"[0, {total})"
                    )
                x, y = np.unravel_index(flat, (nx_, ny_))
                info["x"] = x.astype(np.int64, copy=False)
                info["y"] = y.astype(np.int64, copy=False)
            else:
                info["x"], info["y"] = None, None

        coordinate_parts = [
            np.column_stack((info["x"], info["y"]))
            for info in self.owner._rank_files
            if info["x"] is not None and info["y"] is not None
        ]
        if not coordinate_parts:
            return
        coordinates = np.concatenate(coordinate_parts, axis=0)
        if self.owner.map_shape is not None:
            nx, ny = self.owner.map_shape
            valid = (
                (coordinates[:, 0] >= 0)
                & (coordinates[:, 0] < nx)
                & (coordinates[:, 1] >= 0)
                & (coordinates[:, 1] < ny)
            )
            if not np.all(valid):
                raise ValueError("converted output coordinates fall outside map_shape")
        if np.unique(coordinates, axis=0).shape[0] != coordinates.shape[0]:
            raise ValueError(
                "converted output coordinates contain duplicate (x, y) cells"
            )

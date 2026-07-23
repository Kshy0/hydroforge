"""Rank-file discovery, timeline validation, and coordinate resolution."""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import netCDF4 as nc
import numpy as np

from hydroforge.contracts.temporal import canonical_calendar
from hydroforge.output.netcdf.plan import (
    COMMITTED_STEPS_ATTR, OUTPUT_FORMAT, OUTPUT_VERSION,
)

logger = logging.getLogger(__name__)

class RankOutputCatalog:
    def __init__(self, owner: Any) -> None:
        self.owner = owner

    def _select_coord_name(self, ds: nc.Dataset, saved_points: int) -> Optional[str]:
        """Pick a ('saved_points',) variable to serve as output_coord."""
        if self.owner.coord_name:
            if self.owner.coord_name not in ds.variables:
                raise KeyError(
                    f"requested coordinate {self.owner.coord_name!r} is missing"
                )
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
                if np.ma.isMaskedArray(value) and np.any(
                    np.ma.getmaskarray(value)
                ):
                    continue
                array = np.asarray(value)
                if (
                    array.dtype.kind in "iu"
                    and np.unique(array).size == array.size
                ):
                    candidates.append(name)
        if len(candidates) > 1:
            raise ValueError(
                "multiple integer saved_points coordinates are eligible for "
                f"automatic selection: {candidates}; specify coord_name"
            )
        return candidates[0] if candidates else None

    def _inspect_rank_file(
        self, path: Path, *, expected: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        with nc.Dataset(path, "r") as ds:
            required_attrs = {
                "hydroforge_output_format", "hydroforge_output_version",
                "hydroforge_rank", "hydroforge_world_size",
                COMMITTED_STEPS_ATTR,
            }
            missing_attrs = required_attrs.difference(ds.ncattrs())
            if missing_attrs:
                raise ValueError(
                    f"missing HydroForge output contract attributes: "
                    f"{sorted(missing_attrs)}"
                )
            output_format = ds.getncattr("hydroforge_output_format")
            output_version = ds.getncattr("hydroforge_output_version")
            rank = ds.getncattr("hydroforge_rank")
            world_size = ds.getncattr("hydroforge_world_size")
            if output_format != OUTPUT_FORMAT:
                raise ValueError(
                    f"unsupported output format {output_format!r}"
                )
            if type(output_version) not in {int, np.int32, np.int64} or int(
                output_version,
            ) != OUTPUT_VERSION:
                raise ValueError(
                    f"unsupported output version {output_version!r}"
                )
            if type(rank) not in {int, np.int32, np.int64} or int(rank) < 0:
                raise ValueError("hydroforge_rank must be a non-negative integer")
            if (
                type(world_size) not in {int, np.int32, np.int64}
                or int(world_size) < 1
                or int(rank) >= int(world_size)
            ):
                raise ValueError("invalid hydroforge_world_size/rank contract")
            if self.owner.var_name not in ds.variables:
                raise KeyError(f"variable {self.owner.var_name!r} is missing")
            if "time" not in ds.variables:
                raise KeyError("time variable is missing")
            variable = ds.variables[self.owner.var_name]
            committed = ds.getncattr(COMMITTED_STEPS_ATTR)
            if isinstance(committed, (bool, np.bool_)) or not isinstance(
                committed, (int, np.integer),
            ):
                raise TypeError(f"{COMMITTED_STEPS_ATTR} must be an integer")
            committed = int(committed)
            time_length = len(ds.variables["time"])
            data_length = len(variable)
            if (
                committed < 0
                or committed != time_length
                or data_length != time_length
            ):
                raise ValueError(
                    "rank output has an uncommitted or inconsistent append: "
                    f"committed={committed}, time={time_length}, "
                    f"data={data_length}"
                )
            has_trials = "trial" in ds.dimensions
            has_levels = "levels" in ds.dimensions
            dimensions = (
                ("time", "trial", "saved_points", "levels")
                if has_trials and has_levels else
                ("time", "trial", "saved_points")
                if has_trials else
                ("time", "saved_points", "levels")
                if has_levels else
                ("time", "saved_points")
            )
            if variable.dimensions != dimensions:
                raise ValueError(
                    f"variable {self.owner.var_name!r} has dimensions "
                    f"{variable.dimensions}, expected {dimensions}"
                )
            saved_points = int(ds.dimensions["saved_points"].size)
            metadata = {
                "saved_points": saved_points,
                "has_trials": has_trials,
                "n_trials": (
                    int(ds.dimensions["trial"].size) if has_trials else 0
                ),
                "has_levels": has_levels,
                "n_levels": (
                    int(ds.dimensions["levels"].size) if has_levels else 0
                ),
                "dimensions": variable.dimensions,
                "dtype": np.dtype(variable.dtype),
                "contract_rank": int(rank),
                "world_size": int(world_size),
            }
            if expected is not None and metadata != expected:
                raise ValueError(
                    f"rank output schema differs from {expected}: {metadata}"
                )
            return metadata

    @staticmethod
    def _read_coordinate(
        dataset: nc.Dataset, name: str, saved_points: int,
    ) -> np.ndarray:
        if name not in dataset.variables:
            raise KeyError(f"coordinate {name!r} is missing")
        variable = dataset.variables[name]
        if (
            variable.dimensions != ("saved_points",)
            or len(variable) != saved_points
        ):
            raise ValueError(
                f"coordinate {name!r} must have dimensions ('saved_points',)"
            )
        raw = variable[:]
        if np.ma.isMaskedArray(raw) and np.any(np.ma.getmaskarray(raw)):
            raise ValueError(f"coordinate {name!r} contains missing values")
        coordinate = np.asarray(raw)
        if coordinate.dtype.kind not in "iu":
            raise TypeError(f"coordinate {name!r} must be integer")
        if np.unique(coordinate).size != coordinate.size:
            raise ValueError(f"coordinate {name!r} contains duplicate IDs")
        return coordinate

    def scan(self) -> List[dict]:
        """Locate rank files and collect basic structural metadata."""
        pattern = f"{self.owner.var_name}_rank*.nc"
        files = sorted(self.owner.base_dir.glob(pattern))
        rank_map: Dict[int, List[Tuple[int, Path]]] = {}

        # Regex to match rank and optional year: var_rank0.nc or var_rank0_2000.nc
        if self.owner.split_by_year:
            rank_re = re.compile(
                rf"^{re.escape(self.owner.var_name)}_rank(\d+)_(-?\d+)\.nc$"
            )
        else:
            rank_re = re.compile(rf"^{re.escape(self.owner.var_name)}_rank(\d+)\.nc$")

        for fp in files:
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

            if rank_id not in rank_map:
                rank_map[rank_id] = []
            rank_map[rank_id].append((year, fp))

        if rank_map:
            observed = sorted(rank_map)
            expected = list(range(observed[-1] + 1))
            if observed != expected:
                raise ValueError(
                    "rank output files must form a contiguous set starting at "
                    f"zero: expected {expected}, found {observed}"
                )

        rank_infos: List[dict] = []

        for rank_id in sorted(rank_map.keys()):
            # Sort files by year (or just by name if year is -1, but here we use the tuple)
            # If year is -1, it means no year suffix.
            files_with_year = sorted(rank_map[rank_id], key=lambda x: x[0])
            years = tuple(year for year, _path in files_with_year)
            paths = [path for _year, path in files_with_year]

            # Use the first file to get metadata
            first_fp = paths[0]

            try:
                metadata = self._inspect_rank_file(first_fp)
                if metadata["contract_rank"] != rank_id:
                    raise ValueError(
                        f"file name rank {rank_id} disagrees with contract rank "
                        f"{metadata['contract_rank']}"
                    )
                for path in paths[1:]:
                    self._inspect_rank_file(path, expected=metadata)
                with nc.Dataset(first_fp, "r") as ds:
                    saved_points = metadata["saved_points"]
                    coord_name = self._select_coord_name(ds, saved_points)
                    coord_raw = None
                    if coord_name is not None:
                        coord_raw = self._read_coordinate(
                            ds, coord_name, saved_points,
                        )
                for path in paths[1:]:
                    with nc.Dataset(path, "r") as dataset:
                        observed_name = self._select_coord_name(
                            dataset, saved_points,
                        )
                        if observed_name != coord_name:
                            raise ValueError(
                                f"coordinate name changes from {coord_name!r} "
                                f"to {observed_name!r} in {path.name}"
                            )
                        if coord_name is not None:
                            observed = self._read_coordinate(
                                dataset, coord_name, saved_points,
                            )
                            if (
                                observed.dtype != coord_raw.dtype
                                or not np.array_equal(observed, coord_raw)
                            ):
                                raise ValueError(
                                    f"coordinate {coord_name!r} changes values "
                                    f"or order in {path.name}"
                                )

                rank_infos.append(
                    {
                        "rank_id": rank_id,
                        "years": years,
                        "paths": paths, # List of paths
                        "path": first_fp, # Keep for backward compat / metadata
                        **metadata,
                        "coord_name": coord_name,
                        "coord_raw": coord_raw,
                        "x": None,
                        "y": None,
                    }
                )
            except (OSError, KeyError, TypeError, ValueError) as exc:
                raise RuntimeError(
                    f"Failed to inspect rank {rank_id} file {first_fp}"
                ) from exc

        if rank_infos:
            reference = rank_infos[0]
            for info in rank_infos[1:]:
                for name in (
                    "years", "has_trials", "n_trials", "has_levels",
                    "n_levels", "dimensions", "dtype", "coord_name",
                    "world_size",
                ):
                    if info[name] != reference[name]:
                        raise ValueError(
                            f"rank {info['rank_id']} output {name} differs from "
                            f"rank 0: {info[name]!r} != {reference[name]!r}"
                        )
            world_size = reference["world_size"]
            observed_ranks = [info["rank_id"] for info in rank_infos]
            expected_ranks = list(range(world_size))
            if observed_ranks != expected_ranks:
                raise ValueError(
                    "rank output set is incomplete for declared world_size: "
                    f"expected {expected_ranks}, found {observed_ranks}"
                )
            coordinates = [
                info["coord_raw"] for info in rank_infos
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
          - self.owner._file_time_offsets (list of (start, end) indices for each file in the first rank)
        """
        if not self.owner._rank_files:
            raise RuntimeError("No rank files loaded.")

        master_values = None
        master_datetimes = None
        master_offsets = None
        master_lengths = None
        for info in self.owner._rank_files:
            datetimes = []
            offsets = []
            lengths = []
            current_offset = 0
            calendar = None
            first_units = None
            for path, declared_year in zip(
                info["paths"], info["years"], strict=True,
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
                            f"rank {info['rank_id']} files use inconsistent "
                            "calendars"
                        )
                    raw_values = variable[:]
                    if np.ma.isMaskedArray(raw_values) and np.any(
                        np.ma.getmaskarray(raw_values)
                    ):
                        raise ValueError(
                            f"time variable in {path.name} contains missing values"
                        )
                    values = np.asarray(raw_values)
                    if values.ndim != 1 or values.size == 0:
                        raise ValueError(
                            f"time variable in {path.name} must be non-empty and 1-D"
                        )
                    if values.dtype.kind not in "iuf" or not np.isfinite(
                        values,
                    ).all():
                        raise ValueError(
                            f"time variable in {path.name} must contain finite "
                            "numeric values"
                        )
                    decoded = list(nc.num2date(
                        values, units=units, calendar=file_calendar,
                    ))
                    if self.owner.split_by_year and any(
                        instant.year != declared_year for instant in decoded
                    ):
                        observed_years = sorted({
                            instant.year for instant in decoded
                        })
                        raise ValueError(
                            f"year-split file {path.name} declares year "
                            f"{declared_year} but contains timestamps from "
                            f"{observed_years}"
                        )
                    datetimes.extend(decoded)
                    length = len(decoded)
                    lengths.append(length)
                    offsets.append((current_offset, current_offset + length))
                    current_offset += length
            if any(
                right <= left
                for left, right in zip(datetimes, datetimes[1:])
            ):
                raise ValueError(
                    f"rank {info['rank_id']} output time axis must be strictly "
                    "increasing across files"
                )
            if master_datetimes is None:
                self.owner._time_units = first_units
                self.owner._time_calendar = calendar
                master_datetimes = datetimes
                master_values = np.asarray(nc.date2num(
                    datetimes, self.owner._time_units,
                    self.owner._time_calendar,
                ))
                master_offsets = offsets
                master_lengths = lengths
            else:
                if calendar != self.owner._time_calendar:
                    raise ValueError(
                        f"rank {info['rank_id']} output calendar {calendar!r} "
                        f"differs from rank 0 {self.owner._time_calendar!r}"
                    )
                if lengths != master_lengths:
                    raise ValueError(
                        f"rank {info['rank_id']} output file time lengths "
                        f"{lengths} differ from rank 0 {master_lengths}"
                    )
                observed = np.asarray(nc.date2num(
                    datetimes, self.owner._time_units,
                    self.owner._time_calendar,
                ))
                if not np.array_equal(observed, master_values):
                    raise ValueError(
                        f"rank {info['rank_id']} output timestamps differ from rank 0"
                    )
            info["file_time_offsets"] = tuple(offsets)
        self.owner._file_time_offsets = list(master_offsets)
        self.owner._time_datetimes = list(master_datetimes)
        self.owner._time_values_num = master_values
        self.owner._time_len = len(master_values)

    def compute_coordinates(self, force: bool = False) -> None:
        """Compute (x, y) for each rank (custom converter -> unravel -> None)."""
        for info in self.owner._rank_files:
            if info["coord_raw"] is None or info["saved_points"] == 0:
                info["x"], info["y"] = None, None
                continue
            if (info["x"] is not None and info["y"] is not None) and not force:
                continue

            if self.owner._coord_converter is not None:
                x, y = self.owner._coord_converter(info["coord_raw"])
                x = np.asarray(x)
                y = np.asarray(y)
                expected = (info["saved_points"],)
                if x.shape != expected or y.shape != expected:
                    raise ValueError(
                        f"coordinate converter for {info['path'].name} returned "
                        f"shapes {x.shape}/{y.shape}, expected {expected}"
                    )
                if x.dtype.kind not in "iu" or y.dtype.kind not in "iu":
                    raise TypeError(
                        "coordinate converter must return integer x/y arrays"
                    )
                info["x"], info["y"] = (
                    x.astype(np.int64, copy=False),
                    y.astype(np.int64, copy=False),
                )
                continue

            if self.owner._map_shape is not None:
                nx_, ny_ = self.owner._map_shape
                total = nx_ * ny_
                flat = np.asarray(info["coord_raw"]).astype(np.int64)
                if flat.ndim == 1 and np.all((flat >= 0) & (flat < total)):
                    x, y = np.unravel_index(flat, (nx_, ny_))
                    info["x"] = x.astype(np.int64)
                    info["y"] = y.astype(np.int64)
                else:
                    info["x"], info["y"] = None, None
                    logger.info(
                        "%s output_coord is not a valid linear index; cannot "
                        "auto-convert", info["path"].name,
                    )
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
        if self.owner._map_shape is not None:
            nx, ny = self.owner._map_shape
            valid = (
                (coordinates[:, 0] >= 0) & (coordinates[:, 0] < nx)
                & (coordinates[:, 1] >= 0) & (coordinates[:, 1] < ny)
            )
            if not np.all(valid):
                raise ValueError(
                    "converted output coordinates fall outside map_shape"
                )
        if np.unique(coordinates, axis=0).shape[0] != coordinates.shape[0]:
            raise ValueError(
                "converted output coordinates contain duplicate (x, y) cells"
            )

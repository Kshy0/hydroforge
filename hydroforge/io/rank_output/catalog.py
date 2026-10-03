"""Discovery and validation of the rank files of one output variable."""

from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path

import netCDF4 as nc
import numpy as np

from hydroforge.core.time import DateLike, canonical_calendar
from hydroforge.io.files import SourceFiles
from hydroforge.io.netcdf.encoding import BOOL_LOGICAL_DTYPE, LOGICAL_DTYPE_ATTR
from hydroforge.io.netcdf.read import decoded_dtype
from hydroforge.io.rank_output.schema import (
    POINT_DIM,
    TIME_DIM,
    RankFileHeader,
    parse_rank_file_name,
    read_point_coordinate,
)


@dataclass(frozen=True, slots=True)
class RankLayout:
    """Variable layout and run identity shared by every file of one output."""

    dimensions: tuple[str, ...]
    member_count: int | None
    level_dimension: str | None
    level_count: int
    dtype: np.dtype
    decoded_dtype: np.dtype
    logical_dtype: str | None
    world_size: int
    run_id: str
    coord_name: str | None


@dataclass(frozen=True, slots=True)
class RankFiles:
    """The year-ordered files of one rank and their place on its timeline."""

    rank: int
    paths: tuple[Path, ...]
    time_offsets: tuple[tuple[int, int], ...]
    saved_points: int
    coordinate: np.ndarray | None


@dataclass(frozen=True, slots=True)
class RankCatalog:
    """Validated rank files and the timeline they share."""

    layout: RankLayout
    ranks: tuple[RankFiles, ...]
    time_units: str
    calendar: str
    times: tuple[DateLike, ...]
    time_values: np.ndarray


def _file_layout(
    dataset: nc.Dataset,
    header: RankFileHeader,
    *,
    var_name: str,
    coord_name: str | None,
    path: Path,
) -> tuple[RankLayout, int, np.ndarray | None]:
    """Return the layout, saved-point count and point IDs of one rank file."""

    variable = dataset.variables[var_name]
    logical_dtype = getattr(variable, LOGICAL_DTYPE_ATTR, None)
    if logical_dtype is not None and logical_dtype != BOOL_LOGICAL_DTYPE:
        raise ValueError(
            f"variable {var_name!r} declares unsupported logical dtype "
            f"{logical_dtype!r}"
        )
    # The optional trailing value axis keeps its model-defined name.
    dimensions = tuple(variable.dimensions)
    if not dimensions or dimensions[0] != TIME_DIM:
        raise ValueError(
            f"variable {var_name!r} has dimensions {dimensions}; the first "
            f"dimension must be '{TIME_DIM}'"
        )
    if dataset.variables[TIME_DIM].dimensions != (TIME_DIM,):
        raise ValueError("time variable must have dimensions ('time',)")
    # Data and time share the unlimited time axis and therefore its length.
    time_length = len(dataset.variables[TIME_DIM])
    if header.committed_steps > time_length:
        raise ValueError(
            "rank output has an uncommitted or inconsistent append: "
            f"committed={header.committed_steps}, time={time_length}"
        )
    cursor = 1
    member_count = None
    if cursor < len(dimensions) and dimensions[cursor] == "ensemble":
        member_count = int(dataset.dimensions["ensemble"].size)
        cursor += 1
    if cursor >= len(dimensions) or dimensions[cursor] != POINT_DIM:
        detail = (
            ""
            if POINT_DIM in dimensions
            else "; coordinate-free full outputs have no saved_points "
            "axis and are not supported by MultiRankStatsReader"
        )
        raise ValueError(
            f"variable {var_name!r} has dimensions {dimensions}, expected "
            f"('time', ['ensemble'], 'saved_points', [value_axis]){detail}"
        )
    trailing = dimensions[cursor + 1 :]
    if len(trailing) > 1:
        raise ValueError(
            f"variable {var_name!r} has multiple trailing value dimensions "
            f"{trailing}; the reader's level API supports at most one"
        )
    level_dimension = trailing[0] if trailing else None
    coord_name, coordinate = read_point_coordinate(dataset, name=coord_name, path=path)
    layout = RankLayout(
        dimensions=dimensions,
        member_count=member_count,
        level_dimension=level_dimension,
        level_count=(
            0
            if level_dimension is None
            else int(dataset.dimensions[level_dimension].size)
        ),
        dtype=np.dtype(variable.dtype),
        decoded_dtype=decoded_dtype(variable),
        logical_dtype=logical_dtype,
        world_size=header.world_size,
        run_id=header.run_id,
        coord_name=coord_name,
    )
    return layout, int(dataset.dimensions[POINT_DIM].size), coordinate


@dataclass(frozen=True, slots=True)
class _ScannedRank:
    rank: int
    years: tuple[int | None, ...]
    paths: tuple[Path, ...]
    layout: RankLayout
    saved_points: int
    coordinate: np.ndarray | None
    committed: tuple[int, ...]


def _scan_rank(
    files: SourceFiles,
    rank: int,
    years: tuple[int | None, ...],
    paths: tuple[Path, ...],
    *,
    var_name: str,
    coord_name: str | None,
) -> _ScannedRank:
    """Validate one rank's files against their first file."""

    first: tuple[RankLayout, int] | None = None
    coordinate = None
    committed: list[int] = []
    for path in paths:
        with files.open_netcdf(path) as dataset:
            header = RankFileHeader.read(dataset, path=path)
            if first is None and header.rank != rank:
                raise ValueError(
                    f"file name rank {rank} disagrees with contract rank {header.rank}"
                )
            layout, saved_points, values = _file_layout(
                dataset, header, var_name=var_name, coord_name=coord_name, path=path
            )
        committed.append(header.committed_steps)
        if first is None:
            first, coordinate = (layout, saved_points), values
            continue
        name = layout.coord_name
        if first[0].coord_name != name:
            raise ValueError(
                f"coordinate name changes from {first[0].coord_name!r} to "
                f"{name!r} in {path.name}"
            )
        if name is not None and not np.array_equal(values, coordinate):
            raise ValueError(
                f"coordinate {name!r} changes values or order in {path.name}"
            )
        if (layout, saved_points, header.rank) != (*first, rank):
            raise ValueError(
                f"rank output layout of {path.name} differs from the first file "
                f"of rank {rank}: {layout} != {first[0]}"
            )
    layout, saved_points = first
    return _ScannedRank(
        rank=rank,
        years=years,
        paths=paths,
        layout=layout,
        saved_points=saved_points,
        coordinate=coordinate,
        committed=tuple(committed),
    )


def _rank_timeline(
    files: SourceFiles,
    scanned: _ScannedRank,
) -> tuple[list[DateLike], tuple[tuple[int, int], ...], str, str]:
    """Decode one rank's committed timestamps across its files."""

    rank = scanned.rank
    datetimes: list[DateLike] = []
    offsets: list[tuple[int, int]] = []
    calendar = units = None
    for path, year, steps in zip(
        scanned.paths, scanned.years, scanned.committed, strict=True
    ):
        with files.open_netcdf(path) as dataset:
            variable = dataset.variables[TIME_DIM]
            file_units = getattr(variable, "units", None)
            if not isinstance(file_units, str) or not file_units.strip():
                raise ValueError(f"time variable in {path.name} has no CF units")
            file_calendar = canonical_calendar(
                getattr(variable, "calendar", "standard")
            )
            if units is None:
                units, calendar = file_units, file_calendar
            elif file_calendar != calendar:
                raise ValueError(f"rank {rank} files use inconsistent calendars")
            if np.dtype(variable.dtype).kind not in "iuf":
                raise ValueError(f"time variable in {path.name} must be numeric")
            raw = variable[:steps]
        if np.ma.isMaskedArray(raw) and np.any(np.ma.getmaskarray(raw)):
            raise ValueError(f"time variable in {path.name} contains missing values")
        values = np.asarray(raw)
        if values.ndim != 1:
            raise ValueError(f"time variable in {path.name} must be 1-D")
        if values.dtype.kind not in "iuf" or not np.isfinite(values).all():
            raise ValueError(
                f"time variable in {path.name} must contain finite numeric values"
            )
        decoded = list(nc.num2date(values, units=file_units, calendar=file_calendar))
        if year is not None and any(instant.year != year for instant in decoded):
            observed = sorted({instant.year for instant in decoded})
            raise ValueError(
                f"year-split file {path.name} declares year {year} but contains "
                f"timestamps from {observed}"
            )
        offsets.append((len(datetimes), len(datetimes) + len(decoded)))
        datetimes.extend(decoded)
    if any(right <= left for left, right in zip(datetimes, datetimes[1:])):
        raise ValueError(
            f"rank {rank} output time axis must be strictly increasing across files"
        )
    return datetimes, tuple(offsets), units, calendar


def scan_rank_files(
    files: SourceFiles,
    var_name: str,
    *,
    coord_name: str | None,
    split_by_year: bool,
) -> RankCatalog:
    """Validate the captured candidate files of ``var_name`` as one output.

    Every rank ``0..world_size-1`` must be present with one shared layout,
    run identity and coordinate name; saved-point IDs are unique across
    ranks.  The catalog timeline is the committed prefix common to all ranks.
    """

    by_rank: dict[int, dict[int | None, Path]] = {}
    for path in files.identities:
        parsed = parse_rank_file_name(path.name, var_name, split_by_year=split_by_year)
        if parsed is None:
            raise ValueError(
                f"candidate output file {path.name!r} does not match the "
                f"configured {'year-split' if split_by_year else 'single-file'} "
                "rank naming contract"
            )
        rank, year = parsed
        rank_paths = by_rank.setdefault(rank, {})
        if year in rank_paths:
            raise ValueError(
                f"duplicate output files for rank {rank}"
                + ("" if year is None else f" and year {year}")
            )
        rank_paths[year] = path
    observed = sorted(by_rank)
    if observed != list(range(len(observed))):
        raise ValueError(
            "rank output files must form a contiguous set starting at zero: "
            f"expected {len(observed)} consecutive ranks, found {observed}"
        )

    scanned: list[_ScannedRank] = []
    for rank in observed:
        years = tuple(
            sorted(by_rank[rank], key=lambda year: -1 if year is None else year)
        )
        paths = tuple(by_rank[rank][year] for year in years)
        try:
            scanned.append(
                _scan_rank(
                    files,
                    rank,
                    years,
                    paths,
                    var_name=var_name,
                    coord_name=coord_name,
                )
            )
        except (OSError, KeyError, OverflowError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Failed to inspect rank {rank} file {paths[0]}: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    reference = scanned[0]
    layout = reference.layout
    for other in scanned[1:]:
        if other.years != reference.years:
            raise ValueError(
                f"rank {other.rank} output years differs from rank 0: "
                f"{other.years!r} != {reference.years!r}"
            )
        for field in fields(RankLayout):
            left, right = getattr(other.layout, field.name), getattr(layout, field.name)
            if left != right:
                raise ValueError(
                    f"rank {other.rank} output {field.name} differs from rank 0: "
                    f"{left!r} != {right!r}"
                )
    if len(scanned) != layout.world_size:
        raise ValueError(
            "rank output set is incomplete for declared world_size: expected "
            f"{layout.world_size} ranks, found {len(scanned)}: {observed}"
        )
    coordinates = [
        entry.coordinate for entry in scanned if entry.coordinate is not None
    ]
    if coordinates:
        combined = np.concatenate(coordinates)
        if np.unique(combined).size != combined.size:
            raise ValueError(
                "output coordinates contain duplicate IDs across rank files"
            )

    timelines = [_rank_timeline(files, entry) for entry in scanned]
    common_length = min(len(datetimes) for datetimes, *_rest in timelines)
    if common_length == 0:
        raise ValueError("rank outputs have no common committed time steps")
    datetimes, _offsets, time_units, calendar = timelines[0]
    times = tuple(datetimes[:common_length])
    time_values = np.asarray(nc.date2num(list(times), time_units, calendar))
    for entry, (datetimes, _offsets, _units, other_calendar) in zip(
        scanned[1:], timelines[1:], strict=True
    ):
        if other_calendar != calendar:
            raise ValueError(
                f"rank {entry.rank} output calendar {other_calendar!r} differs "
                f"from rank 0 {calendar!r}"
            )
        observed_values = np.asarray(
            nc.date2num(datetimes[:common_length], time_units, calendar)
        )
        if not np.array_equal(observed_values, time_values):
            raise ValueError(
                f"rank {entry.rank} output timestamps differ from rank 0 within "
                "the common committed prefix"
            )
    for coordinate in coordinates:
        coordinate.setflags(write=False)
    time_values.setflags(write=False)
    return RankCatalog(
        layout=layout,
        ranks=tuple(
            RankFiles(
                rank=entry.rank,
                paths=entry.paths,
                time_offsets=offsets,
                saved_points=entry.saved_points,
                coordinate=entry.coordinate,
            )
            for entry, (_datetimes, offsets, _units, _calendar) in zip(
                scanned, timelines, strict=True
            )
        ),
        time_units=time_units,
        calendar=calendar,
        times=times,
        time_values=time_values,
    )

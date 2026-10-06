# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""CaMa-Flood map readers shared across spatial-mapping producers.

The CaMa map grid is a regular ``(nx, ny)`` geographic grid; the active
catchments are a sparse selection of its cells. These helpers decode the
low-resolution catchment list and the high-resolution (MERIT Hydro) pixels that
each catchment is composed of, returning plain numpy arrays:

* :func:`_read_cama_catchments` -> linear catchment ids on the ``(nx, ny)`` grid.
* :func:`_cama_hires_source_cells` -> per-pixel ``(target row, area, source
  cell)`` used to area-weight a runoff grid onto catchments.
* :func:`_cama_cell_targets` -> the low-resolution cell bounds and areas of
  catchments, for maps used without a high-resolution grid.

Precision strings are validated once by the public mapping builders through
:data:`IndexPrecision` and :data:`FloatPrecision`; the readers here consume
validated values.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, NamedTuple

import numpy as np
from pydantic import AfterValidator

from hydroforge.io.binary import binread, read_map
from hydroforge.mapping.engine import _EARTH_RADIUS_M
from hydroforge.mapping.grid import RegularGrid


def _plain_precision(kind: str, description: str) -> AfterValidator:
    def check(value: str) -> str:
        dtype = np.dtype(value)
        if (
            dtype.kind != kind
            or dtype.hasobject
            or dtype.subdtype is not None
            or dtype.fields is not None
        ):
            raise ValueError(f"must specify a plain {description} dtype")
        return value

    return AfterValidator(check)


IndexPrecision = Annotated[str, _plain_precision("i", "signed integer")]
FloatPrecision = Annotated[str, _plain_precision("f", "floating")]


def _validate_catmxy(catmxy: np.ndarray, *, label: str) -> None:
    if catmxy.dtype.kind != "i":
        raise TypeError(f"{label} must contain signed integer indices")
    x = catmxy[:, :, 0]
    y = catmxy[:, :, 1]
    x_active = x > 0
    y_active = y > 0
    if np.any(x_active != y_active):
        raise ValueError(
            f"{label} must use matching positive 1-based pairs for active pixels"
        )
    inactive = ~x_active
    if np.any(inactive & (x != y)):
        raise ValueError(
            f"{label} inactive pixels must use matching non-positive sentinels"
        )


def _validate_nextxy(nextxy: np.ndarray) -> None:
    """Require the two downstream-pointer components to agree on activity."""

    if nextxy.dtype.kind != "i":
        raise TypeError("nextxy.bin must contain signed integer indices")
    inactive_x = nextxy[:, :, 0] == -9999
    inactive_y = nextxy[:, :, 1] == -9999
    if np.any(inactive_x != inactive_y):
        raise ValueError(
            "nextxy.bin must use (-9999, -9999) for inactive cells; "
            "the downstream x/y components disagree"
        )


def _require_active_catchments(
    nextxy: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    *,
    label: str,
) -> None:
    inactive = nextxy[x, y, 0] == -9999
    if np.any(inactive):
        raise ValueError(
            f"{label} assigns {int(np.count_nonzero(inactive))} high-resolution "
            "pixel(s) to inactive CaMa cells"
        )


def _grid_offset(delta: float, spacing: float, *, label: str) -> int:
    if not np.isfinite(delta):
        raise ValueError(f"{label} must be finite")
    if not np.isfinite(spacing) or spacing <= 0.0:
        raise ValueError(f"{label} grid spacing must be finite and positive")
    quotient = delta / spacing
    if not np.isfinite(quotient) or abs(quotient) > np.iinfo(np.intp).max:
        raise OverflowError(f"{label} grid offset exceeds the platform index range")
    nearest = round(quotient)
    # CaMa metadata stores some grid sizes at only eight decimal places
    # (notably 1/60 degree as 0.01666667).  Across a global axis that textual
    # rounding accumulates to a few thousandths of one cell.  Accept that
    # documented precision loss while still rejecting any material fraction
    # of a cell.
    tolerance = max(
        512.0 * np.finfo(np.float64).eps * max(abs(quotient), 1.0),
        5.0e-7 * max(abs(quotient), 1.0),
    )
    if abs(quotient - nearest) > tolerance:
        raise ValueError(f"{label} is not aligned to grid spacing {spacing!r}")
    return int(nearest)


def _validate_grid_extent(
    lower: float,
    upper: float,
    spacing: float,
    count: int,
    *,
    label: str,
    latitude: bool = False,
) -> None:
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError(f"{label} bounds must be finite and increasing")
    if latitude and (lower < -90.0 or upper > 90.0):
        raise ValueError(f"{label} bounds must lie within [-90, 90]")
    if type(count) is not int or count < 1:
        raise ValueError(f"{label} cell count must be a positive integer")
    observed = _grid_offset(
        upper - lower,
        spacing,
        label=f"{label} extent",
    )
    if observed != count:
        raise ValueError(
            f"{label} extent contains {observed} cells at spacing "
            f"{spacing!r}, expected {count}"
        )


def _read_region_parameters(
    map_dir: Path,
    nx: int,
    ny: int,
) -> tuple[float, float, float, float, float]:
    with open(map_dir / "params.txt") as stream:
        lines = stream.readlines()
    if len(lines) < 8:
        raise ValueError("params.txt must contain at least eight lines")
    gsize = float(lines[3].split()[0])
    west = float(lines[4].split()[0])
    east = float(lines[5].split()[0])
    south = float(lines[6].split()[0])
    north = float(lines[7].split()[0])
    _validate_grid_extent(west, east, gsize, nx, label="CaMa longitude")
    _validate_grid_extent(south, north, gsize, ny, label="CaMa latitude", latitude=True)
    return gsize, west, east, south, north


def _read_hires_tile(
    directory: Path,
    name: str,
    nx: int,
    ny: int,
    *,
    area_precision: str,
    index_precision: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Read one tile's stored km² areas and validate its catchment indices.

    Areas stay in their stored precision; :func:`_pixel_areas_m2` converts
    only the pixels that are used, which keeps a global tile's footprint
    near its file size.
    """
    areas = read_map(
        directory / f"{name}.grdare.bin", (nx, ny), precision=area_precision
    )
    catchments = read_map(
        directory / f"{name}.catmxy.bin", (nx, ny, 2), precision=index_precision
    )
    _validate_catmxy(catchments, label=f"{name}.catmxy.bin")
    active_areas = areas[catchments[:, :, 0] > 0]
    if np.any(~np.isfinite(active_areas) | (active_areas < 0)):
        raise ValueError(
            f"{name}.grdare.bin active pixel areas must be finite and nonnegative"
        )
    return areas, catchments


def _pixel_areas_m2(areas: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Gather stored km² pixel areas as float64 m²."""

    return np.asarray(areas[x, y], dtype=np.float64) * 1e6


def _read_cama_catchments(
    map_dir: Path,
    *,
    lowres_idx_precision: str = "<i4",
) -> tuple[np.ndarray, int, int, np.ndarray]:
    """Read the linear catchment ids from a CaMa map directory.

    Returns ``(catchment_id, nx, ny, nextxy_data)`` where ``catchment_id`` is the
    C-order ``ix*ny+iy`` index of every active cell and ``nextxy_data`` is the
    validated ``(nx, ny, 2)`` downstream-pointer array.
    """
    with open(map_dir / "mapdim.txt") as f:
        lines = f.readlines()
        nx = int(lines[0].split("!!")[0].strip())
        ny = int(lines[1].split("!!")[0].strip())
    if nx < 1 or ny < 1:
        raise ValueError("mapdim.txt grid dimensions must be positive")

    nextxy_data = binread(
        map_dir / "nextxy.bin",
        (nx, ny, 2),
        dtype_str=lowres_idx_precision,
    )
    _validate_nextxy(nextxy_data)
    catchment_x, catchment_y = np.where(nextxy_data[:, :, 0] != -9999)
    catchment_id = np.ravel_multi_index((catchment_x, catchment_y), (nx, ny))
    return catchment_id, nx, ny, nextxy_data


class _HiresTilePixels(NamedTuple):
    """Valid pixels of one tile; pixel ``k`` sits at
    ``(lon[x_index[k]], lat[y_index[k]])``."""

    catchment_id: np.ndarray
    area: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    x_index: np.ndarray
    y_index: np.ndarray


def _lowres_cell_areas(
    map_dir: Path,
    nx: int,
    ny: int,
    x_idx: np.ndarray,
    y_idx: np.ndarray,
    *,
    north: float,
    csize: float,
    map_precision: str,
) -> np.ndarray:
    """Area in m^2 represented by each active CaMa cell.

    ``ctmare.bin`` (unit-catchment area, what CaMa-Flood itself uses to turn
    runoff depth into volume) matches the hires path, whose rows sum the
    hires pixel areas of each catchment.  Without it, the spherical area of
    the low-resolution cell is the closest geometric equivalent.
    """

    catchment_area_path = map_dir / "ctmare.bin"
    if catchment_area_path.exists():
        areas = np.asarray(
            read_map(catchment_area_path, (nx, ny), precision=map_precision),
            dtype=np.float64,
        )[x_idx, y_idx]
        if not np.all(np.isfinite(areas) & (areas > 0.0)):
            raise ValueError(
                "ctmare.bin must hold finite positive unit-catchment areas "
                "(m^2) for every active CaMa cell"
            )
        return areas
    edges = np.clip(north - np.arange(ny + 1, dtype=np.float64) * csize, -90.0, 90.0)
    row_area = (
        _EARTH_RADIUS_M
        * _EARTH_RADIUS_M
        * np.radians(csize)
        * (np.sin(np.radians(edges[:-1])) - np.sin(np.radians(edges[1:])))
    )
    areas = row_area[y_idx]
    if not np.all(np.isfinite(areas) & (areas > 0.0)):
        raise ValueError("active low-resolution cell areas must be finite and positive")
    return areas


def _cama_hires_tiles(
    map_dir: Path,
    nx: int,
    ny: int,
    nextxy_data: np.ndarray,
    *,
    hires_tag: str,
    mapinfo_txt: str,
    hires_idx_precision: str,
    map_precision: str,
) -> list[_HiresTilePixels]:
    """Decode every tile's valid pixels with per-axis coordinates."""

    hires_map_dir = map_dir / hires_tag
    with open(hires_map_dir / mapinfo_txt) as f:
        loc_lines = f.readlines()
    narea = int(loc_lines[0].split()[0])
    if narea < 1:
        raise ValueError(f"{mapinfo_txt} must describe at least one tile")

    if narea == 1:
        data = loc_lines[2].split()
        tile_nx, tile_ny = int(data[6]), int(data[7])
        west, east = float(data[2]), float(data[3])
        south, north = float(data[4]), float(data[5])
        csize = float(data[8])
        _validate_grid_extent(west, east, csize, tile_nx, label="hires longitude")
        _validate_grid_extent(
            south, north, csize, tile_ny, label="hires latitude", latitude=True
        )

        hires_lon = west + (np.arange(tile_nx, dtype=np.float64) + 0.5) * csize
        hires_lat = north - (np.arange(tile_ny, dtype=np.float64) + 0.5) * csize

        tile_name = data[1]
        grid_area, catm = _read_hires_tile(
            hires_map_dir,
            tile_name,
            tile_nx,
            tile_ny,
            area_precision=map_precision,
            index_precision=hires_idx_precision,
        )

        valid = catm[:, :, 0] > 0
        x_idx, y_idx = np.where(valid)
        catm_x = catm[x_idx, y_idx, 0].astype(np.int64, copy=False) - 1
        catm_y = catm[x_idx, y_idx, 1].astype(np.int64, copy=False) - 1
        if np.any(catm_x >= nx) or np.any(catm_y >= ny):
            raise ValueError(
                f"{tile_name}.catmxy.bin contains catchment indices outside "
                f"the ({nx}, {ny}) CaMa grid"
            )
        _require_active_catchments(
            nextxy_data,
            catm_x,
            catm_y,
            label=f"{tile_name}.catmxy.bin",
        )
        catchment_id_hires = np.ravel_multi_index((catm_x, catm_y), (nx, ny))
        return [
            _HiresTilePixels(
                catchment_id_hires,
                _pixel_areas_m2(grid_area, x_idx, y_idx),
                hires_lon,
                hires_lat,
                x_idx,
                y_idx,
            )
        ]

    # --- Multi-tile hires map (catmxy stores global indices) ---
    gsize, reg_west, reg_east, reg_south, reg_north = _read_region_parameters(
        map_dir, nx, ny
    )

    # Regional map is a subset of the global grid starting at (-180, 90).
    global_x_offset = _grid_offset(
        reg_west - (-180.0),
        gsize,
        label="CaMa western global offset",
    )
    global_y_offset = _grid_offset(
        90.0 - reg_north,
        gsize,
        label="CaMa northern global offset",
    )
    csize = float(loc_lines[2].split()[8])
    if not np.isfinite(csize) or csize <= 0.0:
        raise ValueError("hires tile spacing must be finite and positive")

    tiles: list[_HiresTilePixels] = []
    plans = []
    occupied_tiles: list[tuple[str, int, int, int, int]] = []

    for i in range(narea):
        data = loc_lines[2 + i].split()
        tile_name = data[1]
        tw, te = float(data[2]), float(data[3])
        ts, tn = float(data[4]), float(data[5])
        tnx, tny = int(data[6]), int(data[7])
        tile_csize = float(data[8])
        if tile_csize != csize:
            raise ValueError("all hires tiles must use the same grid spacing")
        _validate_grid_extent(tw, te, csize, tnx, label=f"tile {tile_name} longitude")
        _validate_grid_extent(
            ts, tn, csize, tny, label=f"tile {tile_name} latitude", latitude=True
        )

        if te <= reg_west or tw >= reg_east or tn <= reg_south or ts >= reg_north:
            continue

        ix_start = max(
            0,
            _grid_offset(
                reg_west - tw,
                csize,
                label=f"tile {tile_name} western crop",
            ),
        )
        ix_end = min(
            tnx,
            _grid_offset(
                reg_east - tw,
                csize,
                label=f"tile {tile_name} eastern crop",
            ),
        )
        iy_start = max(
            0,
            _grid_offset(
                tn - reg_north,
                csize,
                label=f"tile {tile_name} northern crop",
            ),
        )
        iy_end = min(
            tny,
            _grid_offset(
                tn - reg_south,
                csize,
                label=f"tile {tile_name} southern crop",
            ),
        )
        if ix_end <= ix_start or iy_end <= iy_start:
            continue

        region_x0 = _grid_offset(
            tw + ix_start * csize - reg_west,
            csize,
            label=f"tile {tile_name} regional x origin",
        )
        region_y0 = _grid_offset(
            reg_north - (tn - iy_start * csize),
            csize,
            label=f"tile {tile_name} regional y origin",
        )
        region_x1 = region_x0 + (ix_end - ix_start)
        region_y1 = region_y0 + (iy_end - iy_start)
        for other_name, other_x0, other_x1, other_y0, other_y1 in occupied_tiles:
            if max(region_x0, other_x0) < min(region_x1, other_x1) and max(
                region_y0, other_y0
            ) < min(region_y1, other_y1):
                raise ValueError(
                    f"hires tiles {other_name!r} and {tile_name!r} overlap "
                    "inside the regional CaMa grid"
                )
        occupied_tiles.append(
            (
                tile_name,
                region_x0,
                region_x1,
                region_y0,
                region_y1,
            )
        )

        plans.append((tile_name, tnx, tny, tw, tn, ix_start, ix_end, iy_start, iy_end))

    for tile_name, tnx, tny, tw, tn, ix_start, ix_end, iy_start, iy_end in plans:
        tile_grdare, tile_catmxy = _read_hires_tile(
            hires_map_dir,
            tile_name,
            tnx,
            tny,
            area_precision=map_precision,
            index_precision=hires_idx_precision,
        )

        sub_catmxy = tile_catmxy[ix_start:ix_end, iy_start:iy_end, :]
        sub_grdare = tile_grdare[ix_start:ix_end, iy_start:iy_end]

        sub_lon = tw + (np.arange(ix_start, ix_end, dtype=np.float64) + 0.5) * csize
        sub_lat = tn - (np.arange(iy_start, iy_end, dtype=np.float64) + 0.5) * csize

        valid = sub_catmxy[:, :, 0] > 0
        xi, yi = np.where(valid)
        if len(xi) == 0:
            continue

        vx = sub_catmxy[xi, yi, 0].astype(np.int64, copy=False) - 1 - global_x_offset
        vy = sub_catmxy[xi, yi, 1].astype(np.int64, copy=False) - 1 - global_y_offset
        in_region = (vx >= 0) & (vx < nx) & (vy >= 0) & (vy < ny)
        xi_r, yi_r = xi[in_region], yi[in_region]
        vx_r, vy_r = vx[in_region], vy[in_region]
        if len(xi_r) == 0:
            continue
        _require_active_catchments(
            nextxy_data,
            vx_r,
            vy_r,
            label=f"{tile_name}.catmxy.bin",
        )

        tiles.append(
            _HiresTilePixels(
                np.ravel_multi_index((vx_r, vy_r), (nx, ny)),
                _pixel_areas_m2(sub_grdare, xi_r, yi_r),
                sub_lon,
                sub_lat,
                xi_r,
                yi_r,
            )
        )
    return tiles


def _cama_hires_source_cells(
    map_dir: Path,
    nx: int,
    ny: int,
    nextxy_data: np.ndarray,
    source: RegularGrid,
    *,
    allow_oob: bool,
    target_ids: np.ndarray,
    hires_tag: str = "1min",
    mapinfo_txt: str = "location.txt",
    hires_idx_precision: str = "<i2",
    map_precision: str = "<f4",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode hires pixels directly into target rows and ``source`` cells.

    Each tile's longitude and latitude axes are located on ``source`` once
    and gathered per pixel, so no per-pixel coordinates are built. Returns
    ``(target_row, area_m2, source_index)`` for every pixel whose catchment
    is in ``target_ids``, with ``target_row`` indexing ``target_ids`` and
    ``-1`` source cells for pixels outside the source grid (rejected unless
    ``allow_oob``).
    """
    tiles = _cama_hires_tiles(
        map_dir,
        nx,
        ny,
        nextxy_data,
        hires_tag=hires_tag,
        mapinfo_txt=mapinfo_txt,
        hires_idx_precision=hires_idx_precision,
        map_precision=map_precision,
    )
    order = np.argsort(target_ids, kind="stable")
    sorted_ids = target_ids[order]
    source_nx = source.x.size
    rows: list[np.ndarray] = [np.empty(0, dtype=np.int64)]
    areas: list[np.ndarray] = [np.empty(0, dtype=np.float64)]
    cells: list[np.ndarray] = [np.empty(0, dtype=np.int64)]
    for tile in tiles:
        catchment_id = tile.catchment_id.astype(np.int64, copy=False)
        position = np.minimum(
            np.searchsorted(sorted_ids, catchment_id), max(sorted_ids.size - 1, 0)
        )
        if sorted_ids.size:
            selected = sorted_ids[position] == catchment_id
        else:
            selected = np.zeros(catchment_id.shape, dtype=bool)
        ix = source._x_indices(tile.lon)[tile.x_index[selected]]
        iy = source._y_indices(tile.lat)[tile.y_index[selected]]
        cells.append(np.where((ix >= 0) & (iy >= 0), iy * source_nx + ix, -1))
        rows.append(order[position[selected]].astype(np.int64, copy=False))
        areas.append(tile.area[selected])
    source_index = np.concatenate(cells)
    if not allow_oob:
        bad = int(np.count_nonzero(source_index < 0))
        if bad:
            raise ValueError(
                f"{bad}/{source_index.size} points fall outside the source grid"
            )
    return np.concatenate(rows), np.concatenate(areas), source_index


def _cama_cell_targets(
    map_dir: Path,
    nx: int,
    ny: int,
    target_ids: np.ndarray,
    *,
    map_precision: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Low-resolution cell bounds and areas of CaMa catchments.

    Returns ``(bounds, area_m2)`` per target: ``(west, east, south, north)``
    of each catchment's ``(nx, ny)`` grid cell and the area it represents
    (see :func:`_lowres_cell_areas`). Edges are interpolated between the
    region bounds so the eight-decimal ``params.txt`` cell size does not
    accumulate across the grid.
    """
    csize, west, east, south, north = _read_region_parameters(map_dir, nx, ny)
    x_idx, y_idx = np.unravel_index(target_ids, (nx, ny))
    x_edges = west + (east - west) * (np.arange(nx + 1, dtype=np.float64) / nx)
    y_edges = north + (south - north) * (np.arange(ny + 1, dtype=np.float64) / ny)
    x_edges[-1], y_edges[-1] = east, south
    bounds = np.column_stack(
        (x_edges[x_idx], x_edges[x_idx + 1], y_edges[y_idx + 1], y_edges[y_idx])
    )
    areas = _lowres_cell_areas(
        map_dir,
        nx,
        ny,
        x_idx,
        y_idx,
        north=north,
        csize=csize,
        map_precision=map_precision,
    )
    return bounds, areas

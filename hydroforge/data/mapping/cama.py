"""CaMa-Flood map readers shared across spatial-mapping producers.

The CaMa map grid is a regular ``(nx, ny)`` geographic grid; the active
catchments are a sparse selection of its cells. These helpers decode the
low-resolution catchment list and the high-resolution (MERIT Hydro) pixels that
each catchment is composed of, returning plain numpy arrays:

* :func:`read_cama_catchments` -> linear catchment ids on the ``(nx, ny)`` grid.
* :func:`read_cama_hires_pixels` -> per-pixel ``(catchment_id, area, lon, lat)``
  used to area-weight a runoff grid onto catchments.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np


from hydroforge.data.distributed import binread, read_map


def _binary_precision(
    value: str, *, label: str, kinds: frozenset[str],
) -> np.dtype:
    if type(value) is not str:
        raise TypeError(f"{label} must be an exact dtype string")
    dtype = np.dtype(value)
    if (
        dtype.kind not in kinds
        or dtype.hasobject
        or dtype.subdtype is not None
        or dtype.fields is not None
    ):
        expected = "signed integer" if kinds == frozenset({"i"}) else "floating"
        raise TypeError(f"{label} must specify a plain {expected} dtype")
    return dtype


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
        raise ValueError(
            f"{label} is not aligned to grid spacing {spacing!r}"
        )
    return int(nearest)


def _validate_grid_extent(
    lower: float,
    upper: float,
    spacing: float,
    count: int,
    *,
    label: str,
) -> None:
    if not np.isfinite(lower) or not np.isfinite(upper) or upper <= lower:
        raise ValueError(f"{label} bounds must be finite and increasing")
    if type(count) is not int or count < 1:
        raise ValueError(f"{label} cell count must be a positive integer")
    observed = _grid_offset(
        upper - lower, spacing, label=f"{label} extent",
    )
    if observed != count:
        raise ValueError(
            f"{label} extent contains {observed} cells at spacing "
            f"{spacing!r}, expected {count}"
        )


def _read_region_parameters(
    map_dir: Path, nx: int, ny: int,
) -> tuple[float, float, float, float, float]:
    with open(map_dir / "params.txt", "r") as stream:
        lines = stream.readlines()
    if len(lines) < 8:
        raise ValueError("params.txt must contain at least eight lines")
    gsize = float(lines[3].split()[0])
    west = float(lines[4].split()[0])
    east = float(lines[5].split()[0])
    south = float(lines[6].split()[0])
    north = float(lines[7].split()[0])
    _validate_grid_extent(west, east, gsize, nx, label="CaMa longitude")
    _validate_grid_extent(south, north, gsize, ny, label="CaMa latitude")
    return gsize, west, east, south, north


def read_cama_catchments(
    map_dir: str | Path,
    *,
    lowres_idx_precision: str = "<i4",
) -> tuple[np.ndarray, int, int, np.ndarray]:
    """Read the linear catchment ids from a CaMa map directory.

    Returns ``(catchment_id, nx, ny, nextxy_data)`` where ``catchment_id`` is the
    C-order ``ix*ny+iy`` index of every active cell and ``nextxy_data`` is the
    raw ``(nx, ny, 2)`` downstream-pointer array.
    """
    _binary_precision(
        lowres_idx_precision,
        label="lowres_idx_precision",
        kinds=frozenset({"i"}),
    )
    map_dir = Path(map_dir)
    with open(map_dir / "mapdim.txt", "r") as f:
        lines = f.readlines()
        nx = int(lines[0].split("!!")[0].strip())
        ny = int(lines[1].split("!!")[0].strip())
    if nx < 1 or ny < 1:
        raise ValueError("mapdim.txt grid dimensions must be positive")

    nextxy_data = binread(
        map_dir / "nextxy.bin", (nx, ny, 2),
        dtype_str=lowres_idx_precision,
    )
    _validate_nextxy(nextxy_data)
    catchment_x, catchment_y = np.where(nextxy_data[:, :, 0] != -9999)
    catchment_id = np.ravel_multi_index((catchment_x, catchment_y), (nx, ny))
    return catchment_id, nx, ny, nextxy_data


def read_cama_hires_pixels(
    map_dir: str | Path,
    nx: int,
    ny: int,
    nextxy_data: np.ndarray,
    *,
    hires_tag: str | None = "1min",
    mapinfo_txt: str = "location.txt",
    hires_idx_precision: str = "<i2",
    map_precision: str = "<f4",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode the high-resolution pixels backing each catchment.

    Returns ``(catchment_id_hires, areas, lon, lat)`` with one entry per valid
    high-resolution pixel: the linear catchment id it belongs to (on the
    ``(nx, ny)`` grid), its area in m^2, and its center coordinates.

    When ``hires_tag`` is ``None`` the CaMa grid itself is used as a uniform
    "hires" grid (each active cell maps to itself with unit area).
    """
    _binary_precision(
        hires_idx_precision,
        label="hires_idx_precision",
        kinds=frozenset({"i"}),
    )
    _binary_precision(
        map_precision,
        label="map_precision",
        kinds=frozenset({"f"}),
    )
    if type(nx) is not int or nx < 1 or type(ny) is not int or ny < 1:
        raise ValueError("CaMa grid dimensions must be positive exact integers")
    if np.ma.isMaskedArray(nextxy_data) and np.any(
        np.ma.getmaskarray(nextxy_data)
    ):
        raise ValueError("nextxy_data contains missing values")
    nextxy_data = np.asarray(nextxy_data)
    if nextxy_data.shape != (nx, ny, 2):
        raise ValueError(
            f"nextxy_data must have shape ({nx}, {ny}, 2), "
            f"got {nextxy_data.shape}"
        )
    _validate_nextxy(nextxy_data)
    map_dir = Path(map_dir)

    if hires_tag is None:
        # Use the actual regional CaMa grid as a uniform hires grid.
        csize, west, _east, _south, north = _read_region_parameters(
            map_dir, nx, ny,
        )
        hires_lon = west + (np.arange(nx, dtype=np.float64) + 0.5) * csize
        hires_lat = north - (np.arange(ny, dtype=np.float64) + 0.5) * csize
        x_idx, y_idx = np.where(nextxy_data[:, :, 0] != -9999)
        catchment_id_hires = np.ravel_multi_index((x_idx, y_idx), (nx, ny))
        areas = np.ones(len(x_idx), dtype=np.float64)
        return catchment_id_hires, areas, hires_lon[x_idx], hires_lat[y_idx]

    hires_map_dir = map_dir / hires_tag
    with open(hires_map_dir / mapinfo_txt, "r") as f:
        loc_lines = f.readlines()
    narea = int(loc_lines[0].split()[0])
    if narea < 1:
        raise ValueError(f"{mapinfo_txt} must describe at least one tile")

    if narea == 1:
        data = loc_lines[2].split()
        Nx, Ny = int(data[6]), int(data[7])
        West, East = float(data[2]), float(data[3])
        South, North = float(data[4]), float(data[5])
        csize = float(data[8])
        _validate_grid_extent(West, East, csize, Nx, label="hires longitude")
        _validate_grid_extent(South, North, csize, Ny, label="hires latitude")

        hires_lon = West + (np.arange(Nx, dtype=np.float64) + 0.5) * csize
        hires_lat = North - (np.arange(Ny, dtype=np.float64) + 0.5) * csize

        tile_name = data[1]
        grid_area = np.asarray(read_map(
            hires_map_dir / f"{tile_name}.grdare.bin", (Nx, Ny), precision=map_precision
        ), dtype=np.float64) * 1e6
        catm = read_map(
            hires_map_dir / f"{tile_name}.catmxy.bin", (Nx, Ny, 2), precision=hires_idx_precision
        )
        _validate_catmxy(catm, label=f"{tile_name}.catmxy.bin")

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
        catchment_id_hires = np.ravel_multi_index(
            (catm_x, catm_y), (nx, ny)
        )
        return catchment_id_hires, grid_area[x_idx, y_idx], hires_lon[x_idx], hires_lat[y_idx]

    # --- Multi-tile hires map (catmxy stores global indices) ---
    gsize, reg_west, reg_east, reg_south, reg_north = (
        _read_region_parameters(map_dir, nx, ny)
    )

    # Regional map is a subset of the global grid starting at (-180, 90).
    dXX = _grid_offset(
        reg_west - (-180.0), gsize,
        label="CaMa western global offset",
    )
    dYY = _grid_offset(
        90.0 - reg_north, gsize,
        label="CaMa northern global offset",
    )
    csize = float(loc_lines[2].split()[8])
    if not np.isfinite(csize) or csize <= 0.0:
        raise ValueError("hires tile spacing must be finite and positive")

    all_ids: list[np.ndarray] = []
    all_areas: list[np.ndarray] = []
    all_lon: list[np.ndarray] = []
    all_lat: list[np.ndarray] = []
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
        _validate_grid_extent(ts, tn, csize, tny, label=f"tile {tile_name} latitude")

        if te <= reg_west or tw >= reg_east or tn <= reg_south or ts >= reg_north:
            continue

        ix_start = max(0, _grid_offset(
            reg_west - tw, csize,
            label=f"tile {tile_name} western crop",
        ))
        ix_end = min(tnx, _grid_offset(
            reg_east - tw, csize,
            label=f"tile {tile_name} eastern crop",
        ))
        iy_start = max(0, _grid_offset(
            tn - reg_north, csize,
            label=f"tile {tile_name} northern crop",
        ))
        iy_end = min(tny, _grid_offset(
            tn - reg_south, csize,
            label=f"tile {tile_name} southern crop",
        ))
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
            if (
                max(region_x0, other_x0) < min(region_x1, other_x1)
                and max(region_y0, other_y0) < min(region_y1, other_y1)
            ):
                raise ValueError(
                    f"hires tiles {other_name!r} and {tile_name!r} overlap "
                    "inside the regional CaMa grid"
                )
        occupied_tiles.append((
            tile_name, region_x0, region_x1, region_y0, region_y1,
        ))

        tile_grdare = np.asarray(read_map(
            hires_map_dir / f"{tile_name}.grdare.bin", (tnx, tny), precision=map_precision
        ), dtype=np.float64) * 1e6
        tile_catmxy = read_map(
            hires_map_dir / f"{tile_name}.catmxy.bin", (tnx, tny, 2), precision=hires_idx_precision
        )
        _validate_catmxy(tile_catmxy, label=f"{tile_name}.catmxy.bin")

        sub_catmxy = tile_catmxy[ix_start:ix_end, iy_start:iy_end, :]
        sub_grdare = tile_grdare[ix_start:ix_end, iy_start:iy_end]

        sub_lon = tw + (
            np.arange(ix_start, ix_end, dtype=np.float64) + 0.5
        ) * csize
        sub_lat = tn - (
            np.arange(iy_start, iy_end, dtype=np.float64) + 0.5
        ) * csize

        valid = sub_catmxy[:, :, 0] > 0
        xi, yi = np.where(valid)
        if len(xi) == 0:
            continue

        vx = sub_catmxy[xi, yi, 0].astype(np.int64, copy=False) - 1 - dXX
        vy = sub_catmxy[xi, yi, 1].astype(np.int64, copy=False) - 1 - dYY
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

        all_ids.append(np.ravel_multi_index((vx_r, vy_r), (nx, ny)))
        all_areas.append(sub_grdare[xi_r, yi_r])
        all_lon.append(sub_lon[xi_r])
        all_lat.append(sub_lat[yi_r])

    if not all_ids:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
            np.empty(0, dtype=np.float64),
        )
    return (
        np.concatenate(all_ids),
        np.concatenate(all_areas),
        np.concatenate(all_lon),
        np.concatenate(all_lat),
    )

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


def read_cama_catchments(
    map_dir: str | Path,
    *,
    lowres_idx_precision: str = "<i4",
) -> tuple[np.ndarray, int, int, np.ndarray]:
    """Read the linear catchment ids from a CaMa map directory.

    Returns ``(catchment_id, nx, ny, nextxy_data)`` where ``catchment_id`` is the
    C-order ``ix*ny+iy`` index of every active cell and ``nextxy_data`` is the
    raw ``(nx, ny, 2)`` downstream-pointer array (reused for the uniform-hires
    fallback).
    """
    map_dir = Path(map_dir)
    with open(map_dir / "mapdim.txt", "r") as f:
        lines = f.readlines()
        nx = int(lines[0].split("!!")[0].strip())
        ny = int(lines[1].split("!!")[0].strip())

    nextxy_data = binread(map_dir / "nextxy.bin", (nx, ny, 2), dtype_str=lowres_idx_precision)
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
    map_dir = Path(map_dir)

    if hires_tag is None:
        # Use the CaMa grid itself as a uniform hires grid (global coverage).
        csize = 360.0 / nx
        hires_lon = np.linspace(-180.0 + 0.5 * csize, 180.0 - 0.5 * csize, nx)
        hires_lat = np.linspace(90.0 - 0.5 * csize, -90.0 + 0.5 * csize, ny)
        x_idx, y_idx = np.where(nextxy_data[:, :, 0] != -9999)
        catchment_id_hires = np.ravel_multi_index((x_idx, y_idx), (nx, ny))
        areas = np.ones(len(x_idx), dtype=np.float32)
        return catchment_id_hires, areas, hires_lon[x_idx], hires_lat[y_idx]

    hires_map_dir = map_dir / hires_tag
    with open(hires_map_dir / mapinfo_txt, "r") as f:
        loc_lines = f.readlines()
    narea = int(loc_lines[0].split()[0])

    if narea == 1:
        data = loc_lines[2].split()
        Nx, Ny = int(data[6]), int(data[7])
        West, East = float(data[2]), float(data[3])
        South, North = float(data[4]), float(data[5])
        csize = float(data[8])

        hires_lon = np.linspace(West + 0.5 * csize, East - 0.5 * csize, Nx)
        hires_lat = np.linspace(North - 0.5 * csize, South + 0.5 * csize, Ny)

        tile_name = data[1]
        grid_area = read_map(
            hires_map_dir / f"{tile_name}.grdare.bin", (Nx, Ny), precision=map_precision
        ) * 1e6
        catm = read_map(
            hires_map_dir / f"{tile_name}.catmxy.bin", (Nx, Ny, 2), precision=hires_idx_precision
        )

        valid = catm[:, :, 0] > 0
        x_idx, y_idx = np.where(valid)
        catm -= 1  # 1-based -> 0-based
        catchment_id_hires = np.ravel_multi_index(
            (catm[x_idx, y_idx, 0], catm[x_idx, y_idx, 1]), (nx, ny)
        )
        return catchment_id_hires, grid_area[x_idx, y_idx], hires_lon[x_idx], hires_lat[y_idx]

    # --- Multi-tile hires map (catmxy stores global indices) ---
    with open(map_dir / "params.txt", "r") as f:
        plines = f.readlines()
    gsize = float(plines[3].split()[0])
    reg_west = float(plines[4].split()[0])
    reg_east = float(plines[5].split()[0])
    reg_south = float(plines[6].split()[0])
    reg_north = float(plines[7].split()[0])

    # Regional map is a subset of the global grid starting at (-180, 90).
    dXX = int(round((reg_west - (-180.0)) / gsize))
    dYY = int(round((90.0 - reg_north) / gsize))
    csize = float(loc_lines[2].split()[8])

    all_ids: list[np.ndarray] = []
    all_areas: list[np.ndarray] = []
    all_lon: list[np.ndarray] = []
    all_lat: list[np.ndarray] = []

    for i in range(narea):
        data = loc_lines[2 + i].split()
        tile_name = data[1]
        tw, te = float(data[2]), float(data[3])
        ts, tn = float(data[4]), float(data[5])
        tnx, tny = int(data[6]), int(data[7])

        if te <= reg_west or tw >= reg_east or tn <= reg_south or ts >= reg_north:
            continue

        ix_start = max(0, int(round((reg_west - tw) / csize)))
        ix_end = min(tnx, int(round((reg_east - tw) / csize)))
        iy_start = max(0, int(round((tn - reg_north) / csize)))
        iy_end = min(tny, int(round((tn - reg_south) / csize)))
        if ix_end <= ix_start or iy_end <= iy_start:
            continue

        tile_grdare = read_map(
            hires_map_dir / f"{tile_name}.grdare.bin", (tnx, tny), precision=map_precision
        ) * 1e6
        tile_catmxy = read_map(
            hires_map_dir / f"{tile_name}.catmxy.bin", (tnx, tny, 2), precision=hires_idx_precision
        )

        sub_catmxy = tile_catmxy[ix_start:ix_end, iy_start:iy_end, :]
        sub_grdare = tile_grdare[ix_start:ix_end, iy_start:iy_end]

        sub_lon = np.linspace(
            tw + (ix_start + 0.5) * csize, tw + (ix_end - 0.5) * csize, ix_end - ix_start
        )
        sub_lat = np.linspace(
            tn - (iy_start + 0.5) * csize, tn - (iy_end - 0.5) * csize, iy_end - iy_start
        )

        valid = sub_catmxy[:, :, 0] > 0
        xi, yi = np.where(valid)
        if len(xi) == 0:
            continue

        vx = sub_catmxy[xi, yi, 0].astype(np.int32) - 1 - dXX
        vy = sub_catmxy[xi, yi, 1].astype(np.int32) - 1 - dYY
        in_region = (vx >= 0) & (vx < nx) & (vy >= 0) & (vy < ny)
        xi_r, yi_r = xi[in_region], yi[in_region]
        vx_r, vy_r = vx[in_region], vy[in_region]
        if len(xi_r) == 0:
            continue

        all_ids.append(np.ravel_multi_index((vx_r, vy_r), (nx, ny)))
        all_areas.append(sub_grdare[xi_r, yi_r])
        all_lon.append(sub_lon[xi_r])
        all_lat.append(sub_lat[yi_r])

    return (
        np.concatenate(all_ids),
        np.concatenate(all_areas),
        np.concatenate(all_lon),
        np.concatenate(all_lat),
    )

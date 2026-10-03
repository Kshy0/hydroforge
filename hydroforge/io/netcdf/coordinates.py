"""CF coordinate axes and their cell bounds as stored in NetCDF files."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

LONGITUDE_NAMES = ("lon", "longitude", "long", "x")
LATITUDE_NAMES = ("lat", "latitude", "y")


@dataclass(frozen=True, slots=True)
class CoordinateAxis:
    """One coordinate variable with its stored dtype, CF attributes and bounds."""

    name: str
    values: np.ndarray
    units: str | None
    standard_name: str | None
    bounds: np.ndarray | None


def _unmasked(raw: Any, *, message: str) -> np.ndarray:
    if np.ma.isMaskedArray(raw) and np.any(np.ma.getmaskarray(raw)):
        raise ValueError(message)
    return np.asarray(raw)


def read_axis(
    dataset: Any,
    names: Sequence[str],
    *,
    dimension: str | None = None,
    path: Path,
    label: str,
) -> CoordinateAxis:
    """Read the unique coordinate variable named in ``names``.

    With ``dimension`` the candidates are ``dimension`` itself and ``names``
    whose dimensions are exactly ``(dimension,)``; otherwise any variable in
    ``names``, one- or two-dimensional (a rectilinear coordinate grid).  CF
    bounds are the ``bounds`` attribute or ``<name>_bnds`` in the
    coordinate's group, read only for one-dimensional coordinates.
    """

    if dimension is None:
        candidates = [
            dataset.variables[name]
            for name in dict.fromkeys(names)
            if name in dataset.variables
        ]
        where = ""
    else:
        candidates = [
            dataset.variables[name]
            for name in dict.fromkeys((dimension, *names))
            if name in dataset.variables
            and dataset.variables[name].dimensions == (dimension,)
        ]
        where = f" for dimension {dimension!r}"
    if not candidates:
        if dimension is None:
            raise ValueError(f"None of {tuple(names)!r} found in {path.name}")
        raise ValueError(
            f"Unable to find a one-dimensional {label} coordinate{where} in {path.name}"
        )
    if len(candidates) > 1:
        raise ValueError(
            f"Ambiguous {label} coordinates{where} in {path.name}: "
            f"{[variable.name for variable in candidates]}"
        )
    variable = candidates[0]
    if variable.ndim not in {1, 2}:
        raise ValueError(
            f"{label} coordinate {variable.name!r} in {path.name} must be one- "
            "or two-dimensional"
        )
    units = None
    if "units" in variable.ncattrs():
        units = variable.getncattr("units")
        if not isinstance(units, str) or not units.strip():
            raise ValueError(
                f"{label} coordinate {variable.name!r} in {path.name} "
                "must define units as a non-empty string when present"
            )
    bounds = (
        _read_bounds(variable, label=label, path=path) if variable.ndim == 1 else None
    )
    values = _unmasked(
        variable[:],
        message=(
            f"{label} coordinate {variable.name!r} in {path.name} contains "
            "missing values"
        ),
    )
    standard_name = getattr(variable, "standard_name", None)
    return CoordinateAxis(
        name=variable.name,
        values=values,
        units=units,
        standard_name=None if standard_name is None else str(standard_name),
        bounds=bounds,
    )


def _read_bounds(variable: Any, *, label: str, path: Path) -> np.ndarray | None:
    declared = getattr(variable, "bounds", None)
    if declared is not None and (not isinstance(declared, str) or not declared.strip()):
        raise ValueError(
            f"{label} coordinate {variable.name!r} in {path} has invalid bounds "
            f"reference {declared!r}"
        )
    name = declared.strip() if declared is not None else f"{variable.name}_bnds"
    group = variable.group()
    if name not in group.variables:
        if declared is not None:
            raise ValueError(
                f"{label} coordinate {variable.name!r} in {path} declares "
                f"unresolved bounds {declared!r}"
            )
        return None
    bounds = group.variables[name]
    expected = (variable.shape[0], 2)
    if bounds.shape != expected:
        raise ValueError(
            f"{label} bounds in {path.name} must have shape {expected}, got {bounds.shape}"
        )
    values = _unmasked(
        bounds[:], message=f"{label} bounds in {path.name} contain missing values"
    )
    return values

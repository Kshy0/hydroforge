"""Deterministic output-file planning independent from NetCDF mutation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from hydroforge.serialization.netcdf import (
    COMMITTED_STEPS_ATTR as COMMITTED_STEPS_ATTR,
    OUTPUT_FORMAT as OUTPUT_FORMAT,
    OUTPUT_VERSION as OUTPUT_VERSION,
    RUN_ID_ATTR as RUN_ID_ATTR,
)


@dataclass(frozen=True, slots=True)
class OutputFilePlan:
    directory: Path
    variable: str
    rank: int
    year: int | None

    @property
    def path(self) -> Path:
        suffix = f"_rank{self.rank}"
        if self.year is not None:
            suffix += f"_{self.year}"
        return self.directory / f"{self.variable}{suffix}.nc"


@dataclass(frozen=True, slots=True)
class NetCDFCreateRequest:
    variable: str
    schema: Any
    coordinate_values: Any
    output_dir: Path
    rank: int
    world_size: int
    year: int | None
    calendar: str
    time_unit: str
    static_variables: Mapping[str, Mapping[str, Any]]
    run_id: str | None = None


@dataclass(frozen=True, slots=True)
class NetCDFWriteRequest:
    variable: str
    data: np.ndarray
    output_path: Path
    times: tuple[Any, ...]

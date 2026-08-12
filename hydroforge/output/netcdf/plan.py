"""Deterministic output-file planning independent from NetCDF mutation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np


OUTPUT_FORMAT = "hydroforge.statistics"
OUTPUT_VERSION = 3
COMMITTED_STEPS_ATTR = "hydroforge_committed_steps"
RUN_ID_ATTR = "hydroforge_run_id"

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
    metadata: Mapping[str, Any]
    coordinate_values: Any
    output_dir: Path
    rank: int
    world_size: int
    year: int | None
    calendar: str
    time_unit: str
    num_trials: int
    static_variables: Mapping[str, Mapping[str, Any]]
    run_id: str | None = None
    netcdf_options: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NetCDFWriteRequest:
    variable: str
    data: np.ndarray
    output_path: Path
    times: tuple[Any, ...]

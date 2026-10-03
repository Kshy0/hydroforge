"""NetCDF storage shared by gridded and exported datasets.

Both kinds bind a provisional calendar to the files' CF calendar, scan only the
files their plan needs, size automatic chunks from the physical layout, and
finish reads with the same aggregation, unit-factor and narrowing rule.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
from pydantic import AfterValidator, Field

from hydroforge.core.arrays import positive_finite_float64
from hydroforge.core.time import DateLike
from hydroforge.core.validation import FrozenMapping
from hydroforge.data.datasets.plan import TemporalDomain
from hydroforge.data.datasets.timeline import (
    NetCDFTimeline,
    StorageLayout,
    TimelineScan,
    probe_calendar,
)
from hydroforge.data.datasets.values import AggregationMethod, convert
from hydroforge.io.files import FileInspection, SourceFiles

if TYPE_CHECKING:
    from hydroforge.data.datasets.exported import ExportedDataset
    from hydroforge.data.datasets.netcdf import NetCDFDataset

SOURCE_FILE_LABEL = "Dataset source file"
UnitFactor = Annotated[
    float, AfterValidator(partial(positive_finite_float64, label="unit_factor"))
]
TimeAggregation = (
    AggregationMethod
    | Annotated[
        FrozenMapping[Annotated[str, Field(min_length=1)], AggregationMethod],
        Field(min_length=1),
    ]
    | None
)


@dataclass(frozen=True, slots=True)
class NetCDFStore:
    """Identities, layout and compiled timeline of one NetCDF variable."""

    files: SourceFiles
    layout: StorageLayout
    timeline: NetCDFTimeline
    unit_factor: float
    aggregation: AggregationMethod | Mapping[str, AggregationMethod] | None

    def path(self, key: str) -> Path:
        return self.layout.path(key)

    def finish(
        self, values: np.ndarray, *, out_dtype: str, label: str
    ) -> np.ndarray | dict[str, np.ndarray]:
        """Aggregate, convert units and narrow one ingested read."""

        return convert(
            values,
            out_dtype=out_dtype,
            unit_factor=self.unit_factor,
            aggregation=self.aggregation,
            factor=self.timeline.aggregation_factor,
            label=label,
        )


def concatenate_reads(blocks: list[np.ndarray]) -> np.ndarray:
    """Join per-file reads on the time axis, keeping any missing-value mask."""

    if len(blocks) == 1:
        return blocks[0]
    if any(np.ma.isMaskedArray(block) for block in blocks):
        return np.ma.concatenate(blocks, axis=0)
    return np.concatenate(blocks, axis=0)


def _storage_times(domain: TemporalDomain, offset: timedelta) -> list[DateLike]:
    """Distinct storage times of the spin-up and main samples, in read order."""

    times = dict.fromkeys(
        time + offset for time in (*domain.spinup_times(), *domain.times())
    )
    return list(times)


def scan_storage(
    dataset: NetCDFDataset | ExportedDataset,
    domain: TemporalDomain,
    layout: StorageLayout,
    inspection: FileInspection,
    *,
    storage_offset: timedelta,
    inspect_variable: Callable[[Any, Path], None],
) -> tuple[TimelineScan, TemporalDomain]:
    """Bind the domain to the storage calendar and scan the files it needs.

    A provisional (undeclared) calendar is replaced by the CF calendar of the
    first existing file; a declared calendar must match it.  With time
    aggregation each output time needs the records of its interval.
    """

    windows = getattr(dataset, "window_starts", None)
    length = getattr(dataset, "window_length", None)

    def validate_windows() -> None:
        if windows is not None and int(windows[-1]) + length > domain.count:
            raise ValueError(
                "window_starts and window_length extend beyond the main "
                f"source axis of {domain.count} steps"
            )

    if domain.calendar_declared:
        validate_windows()
    width = None if dataset.time_aggregation is None else dataset.time_interval
    required = _storage_times(domain, storage_offset)
    calendar = probe_calendar(
        inspection, layout, dataset.var_name, required, support_width=width
    )
    if calendar is not None and calendar != domain.calendar:
        if domain.calendar_declared:
            raise ValueError(
                f"forcing files use calendar {calendar!r}, but the dataset "
                f"declares or implies calendar {domain.calendar!r}"
            )
        domain = dataset._declared_domain(calendar)
        required = _storage_times(domain, storage_offset)
    if not domain.calendar_declared:
        validate_windows()
    scan = TimelineScan(
        inspection,
        layout,
        variable=dataset.var_name,
        calendar=domain.calendar,
        template=type(domain.start),
        inspect_variable=inspect_variable,
    )
    scan.scan(required, support_width=width)
    return scan, domain


def storage_chunk_len(
    declared: int | None,
    domain: TemporalDomain,
    layout: StorageLayout,
    inspection: FileInspection,
    scan: TimelineScan,
    *,
    storage_offset: timedelta,
    plan: Callable[[Path], int],
) -> int:
    """The declared chunk length, or one planned from the first file's layout.

    The storage planners count source records; each aggregated output step
    reads ``aggregation_factor`` of them.
    """

    if declared is not None:
        return declared
    first = layout.path(layout.keys()(domain.start + storage_offset))
    with inspection.open(first) as path:
        records = plan(path)
    factor = scan.aggregation_factor
    return max(1, records // factor) if factor > 1 else records

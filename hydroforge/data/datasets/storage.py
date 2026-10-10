# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""NetCDF storage shared by gridded and exported datasets.

Both kinds bind a provisional calendar to the files' CF calendar, scan only the
files their plan needs, size automatic chunks from the physical layout, and
finish reads with the same aggregation, unit-factor and narrowing rule.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any

import numpy as np
from pydantic import AfterValidator, Field

from hydroforge.core.arrays import positive_finite_float64
from hydroforge.core.time import DateLike, calendars_equivalent
from hydroforge.core.units import check_units, normalize_units
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
UnitsName = Annotated[str, Field(min_length=1)]
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
    unit_scale: float = 1.0
    unit_offset: float = 0.0

    def path(self, key: str) -> Path:
        return self.layout.path(key)

    @property
    def converts_units(self) -> bool:
        return (self.unit_factor, self.unit_scale, self.unit_offset) != (1.0, 1.0, 0.0)

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
            unit_scale=self.unit_scale,
            unit_offset=self.unit_offset,
        )


def variable_units(variable: Any) -> str | None:
    """The ``units`` attribute of a NetCDF variable, if it records one."""

    units = getattr(variable, "units", None)
    if units is None:
        return None
    return units if isinstance(units, str) else str(units)


def resolve_units(
    dataset: Any, file_units: Iterable[str | None] = ()
) -> tuple[float, float, float]:
    """Return the ``(unit_factor, unit_scale, unit_offset)`` a dataset applies.

    Without ``target_units`` the declared ``unit_factor`` applies unchanged.
    With it, the source units are ``source_units`` or else the variable's
    ``units`` attribute (``file_units`` holds one entry per inspected file),
    and :func:`~hydroforge.core.units.check_units` converts them to
    ``target_units``; an explicitly declared ``unit_factor`` is the explicit
    factor (``1 / unit_factor``) of a pair the table does not know.
    """

    name = getattr(dataset, "var_name", type(dataset).__name__)
    label = f"{type(dataset).__name__} {name!r}"
    declared = dataset.source_units
    if dataset.target_units is None:
        if declared is not None:
            raise ValueError(f"{label}: source_units requires target_units")
        return dataset.unit_factor, 1.0, 0.0
    recorded = {units for units in file_units}
    spelled = {normalize_units(units) for units in recorded if units is not None}
    if declared is not None:
        conflicting = spelled.difference({normalize_units(declared)})
        if conflicting:
            raise ValueError(
                f"{label}: source_units={declared!r} disagrees with the "
                f"variable's units attribute {sorted(conflicting)}"
            )
        source = declared
    else:
        if None in recorded or not recorded:
            raise ValueError(
                f"{label}: the source variable has no units attribute; pass "
                "source_units to convert to target_units"
            )
        if len(spelled) > 1:
            raise ValueError(
                f"{label}: source files record different units {sorted(spelled)}"
            )
        source = next(iter(recorded - {None}))
    explicit = (
        1.0 / dataset.unit_factor if "unit_factor" in dataset.model_fields_set else None
    )
    try:
        scale, offset = check_units(source, dataset.target_units, explicit)
    except ValueError as error:
        raise ValueError(f"{label}: {error}") from None
    aggregation = getattr(dataset, "time_aggregation", None)
    methods = (
        set(aggregation.values()) if isinstance(aggregation, Mapping) else {aggregation}
    )
    if offset != 0.0 and "sum" in methods:
        # A sum of n shifted values carries n offsets; no single affine
        # conversion of the sum is correct.
        raise ValueError(
            f"{label}: the offset unit conversion from {source!r} to "
            f"{dataset.target_units!r} cannot be combined with "
            "time_aggregation='sum'; aggregate with 'mean'/'min'/'max' or "
            "store the source in an offset-free unit"
        )
    return 1.0, scale, offset


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
    first existing file; a declared calendar must match it, where standard and
    proleptic Gregorian match each other after 1582-10-15.  With time
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
    # A declared standard calendar also reads proleptic Gregorian files (and
    # vice versa): both label every post-1582 date identically.
    if (
        calendar is not None
        and calendar != domain.calendar
        and not (
            domain.calendar_declared and calendars_equivalent(calendar, domain.calendar)
        )
    ):
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
        reference=domain.start,
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

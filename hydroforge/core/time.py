# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Calendar-aware dates and exact integer durations shared by every layer."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import datetime, timedelta
from typing import Annotated, Any, TypeAlias

import cftime
from pydantic import Field, validate_call

from hydroforge.core.validation import HydroForgeModel

DateLike: TypeAlias = datetime | cftime.datetime


_CALENDAR_ALIASES = {
    "gregorian": "standard",
    "standard": "standard",
    "365_day": "noleap",
    "366_day": "all_leap",
}

_MICROSECONDS_PER_SECOND = 1_000_000
_SECONDS_PER_DAY = 86_400

_CFTIME_DATETIME_TYPES = {
    "proleptic_gregorian": cftime.DatetimeProlepticGregorian,
    "noleap": cftime.DatetimeNoLeap,
    "all_leap": cftime.DatetimeAllLeap,
    "360_day": cftime.Datetime360Day,
    "julian": cftime.DatetimeJulian,
}


def timedelta_microseconds(value: timedelta) -> int:
    """Return the exact integer duration represented by ``timedelta``.

    ``timedelta.total_seconds()`` is a float and loses microseconds for long
    spans.  Temporal alignment, counts, and persisted identities must use this
    definition; floats are reserved for values handed to numerical kernels.
    """

    return (
        value.days * _SECONDS_PER_DAY + value.seconds
    ) * _MICROSECONDS_PER_SECOND + value.microseconds


_Label = Annotated[str, Field(min_length=1)]


@validate_call(config=HydroForgeModel.model_config)
def timedelta_quotient(
    duration: timedelta,
    interval: timedelta,
    *,
    duration_label: _Label = "duration",
    interval_label: _Label = "interval",
) -> int:
    """Return the exact integral ratio of two durations."""

    numerator = timedelta_microseconds(duration)
    denominator = timedelta_microseconds(interval)
    if denominator <= 0:
        raise ValueError(f"{interval_label} must be positive")
    quotient, remainder = divmod(numerator, denominator)
    if remainder:
        raise ValueError(
            f"{duration_label}={duration!r} is not an exact multiple of "
            f"{interval_label}={interval!r}"
        )
    return quotient


def canonical_calendar(calendar: str) -> str:
    """Normalize only aliases that cftime defines as equivalent."""
    if not isinstance(calendar, str) or not calendar.strip():
        raise ValueError("calendar must be a non-empty string")
    normalized = calendar.strip().lower()
    canonical = _CALENDAR_ALIASES.get(normalized, normalized)
    if canonical != "standard" and canonical not in _CFTIME_DATETIME_TYPES:
        known = sorted({*_CALENDAR_ALIASES, *_CFTIME_DATETIME_TYPES})
        raise ValueError(f"unknown calendar {calendar!r}; expected one of {known}")
    return canonical


def _calendar_date(
    value: DateLike, calendar: str, has_year_zero: bool | None = None
) -> DateLike:
    """Rebuild a checked date in one canonical calendar's representation."""

    components = (
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second,
        value.microsecond,
    )
    if calendar == "standard":
        return datetime(*components)
    try:
        date_type = _CFTIME_DATETIME_TYPES[calendar]
    except KeyError as error:
        raise ValueError(f"unsupported simulation calendar {calendar!r}") from error
    try:
        return date_type(*components, has_year_zero=has_year_zero)
    except ValueError as error:
        raise ValueError(
            f"date {value!r} cannot be represented in calendar {calendar!r}"
        ) from error


def normalize_calendar_dates(
    values: Mapping[str, DateLike | None],
    *,
    calendar: str | None,
    preserve_cftime_declaration: bool = False,
) -> tuple[str, dict[str, DateLike | None], bool]:
    """Infer one calendar and bind plain ``datetime`` values to it.

    Python ``datetime`` carries no CF-calendar declaration and is therefore
    treated as a civil date whose intended calendar may be supplied by an
    explicit ``calendar`` argument, another cftime bound, or storage metadata.
    A cftime value does carry a declaration; conflicting cftime calendars are
    never reinterpreted.

    The returned boolean is true only when neither the caller nor any cftime
    value selected the calendar, so a storage-backed owner may still replace
    the provisional ``standard`` default after inspecting its time axis.
    Nested declarations may request preservation of a standard-calendar
    cftime representation until an enclosing schedule or file binds it.
    """

    observed: dict[str, list[str]] = {}
    for label, value in values.items():
        if value is None:
            continue
        require_date(value, label=label)
        if isinstance(value, cftime.datetime):
            observed.setdefault(date_calendar(value), []).append(label)
    if len(observed) > 1:
        detail = ", ".join(
            f"{name!r} from {labels}" for name, labels in sorted(observed.items())
        )
        raise ValueError(f"datetime values use conflicting calendars: {detail}")

    configured = None if calendar is None else canonical_calendar(calendar)
    inferred = next(iter(observed), None)
    if configured is not None and inferred is not None and configured != inferred:
        labels = observed[inferred]
        raise ValueError(
            f"calendar {configured!r} conflicts with cftime calendar "
            f"{inferred!r} from {labels}"
        )
    resolved = configured or inferred or "standard"
    standard_cftime = resolved == "standard" and (
        preserve_cftime_declaration
        and any(isinstance(value, cftime.datetime) for value in values.values())
        or any(
            value is not None
            and (
                value.year > 9999
                or (value.year, value.month, value.day) < (1582, 10, 15)
            )
            for value in values.values()
        )
    )
    template = next(
        (value for value in values.values() if isinstance(value, cftime.datetime)), None
    )
    normalized: dict[str, DateLike | None] = {}
    for label, value in values.items():
        if value is None:
            normalized[label] = None
            continue
        try:
            if standard_cftime:
                normalized[label] = cftime.DatetimeGregorian(
                    value.year,
                    value.month,
                    value.day,
                    value.hour,
                    value.minute,
                    value.second,
                    value.microsecond,
                    has_year_zero=getattr(
                        value,
                        "has_year_zero",
                        getattr(template, "has_year_zero", False),
                    ),
                )
            else:
                normalized[label] = _calendar_date(
                    value,
                    resolved,
                    getattr(
                        value, "has_year_zero", getattr(template, "has_year_zero", None)
                    ),
                )
        except (TypeError, ValueError, OverflowError) as error:
            raise ValueError(
                f"{label} cannot be represented in calendar {resolved!r}: {value!r}"
            ) from error
    return resolved, normalized, configured is None and inferred is None


def date_calendar(value: DateLike) -> str:
    calendar = getattr(value, "calendar", None)
    if calendar is not None:
        return canonical_calendar(calendar)
    if isinstance(value, datetime):
        return "standard"
    raise ValueError("calendar value must be datetime or cftime.datetime")


def require_calendar(value: DateLike, expected: str, *, label: str) -> None:
    observed = date_calendar(value)
    expected = canonical_calendar(expected)
    if observed != expected:
        raise ValueError(f"{label} uses calendar {observed!r}, expected {expected!r}")


def require_date(value: Any, *, label: str) -> None:
    """Reject values that are not timezone-naive datetime or cftime dates."""

    if not isinstance(value, (datetime, cftime.datetime)):
        raise ValueError(f"{label} must be a datetime value")
    if isinstance(value, datetime) and value.tzinfo is not None:
        raise ValueError(
            f"{label} must be timezone-naive; simulation calendars cannot "
            "mix wall-clock offsets with calendar arithmetic"
        )

# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Unit spellings and the explicit conversion table of forcing values.

``normalize_units`` turns common spellings into one canonical form
(``"kg/m2/s"``, ``"kg m**-2 s**-1"`` and ``"kg m-2 s-1"`` all become
``"kg m-2 s-1"``). ``check_units`` returns the ``(scale, offset)`` with
``target = source * scale + offset`` from a deliberately small table; it
never guesses: an unknown pair is an error unless the caller passes an
explicit ``factor``.
"""

from __future__ import annotations

import math
import re

# Whole-string aliases, matched case-insensitively after whitespace collapse.
_ALIASES: dict[str, str] = {
    "degc": "degC",
    "deg c": "degC",
    "deg_c": "degC",
    "degree_c": "degC",
    "degrees_c": "degC",
    "degree_celsius": "degC",
    "degrees_celsius": "degC",
    "degrees celsius": "degC",
    "celsius": "degC",
    "°c": "degC",
    "k": "K",
    "kelvin": "K",
    "degk": "K",
    "deg_k": "K",
    "degrees_k": "K",
    "%": "%",
    "percent": "%",
    "1": "1",
    "-": "1",
    "": "1",
    "fraction": "1",
    "dimensionless": "1",
    "unitless": "1",
    "pa": "Pa",
    "hpa": "hPa",
    "mbar": "hPa",
    "mb": "hPa",
    "millibar": "hPa",
    "kpa": "kPa",
}

# Symbol spellings inside a product of terms.
_SYMBOLS: dict[str, str] = {
    "d": "day",
    "day": "day",
    "days": "day",
    "h": "h",
    "hr": "h",
    "hour": "h",
    "hours": "h",
    "s": "s",
    "sec": "s",
    "second": "s",
    "seconds": "s",
    "m": "m",
    "meter": "m",
    "meters": "m",
    "metre": "m",
    "metres": "m",
    "mm": "mm",
    "cm": "cm",
    "km": "km",
    "kg": "kg",
    "g": "g",
    "w": "W",
    "pa": "Pa",
    "hpa": "hPa",
    "kpa": "kPa",
    "l": "L",
}

# Canonical position of a symbol within the positive or negative terms.
_ORDER: dict[str, int] = {
    **dict.fromkeys(("mm", "cm", "m", "km", "L"), 1),
    **dict.fromkeys(("s", "h", "day"), 2),
}

_TERM = re.compile(r"^([A-Za-z°%]+)(?:\^|\*\*)?([+-]?\d+)?$")

# Families: canonical unit -> (scale, offset) to the family's base unit.
_DAY = 86400.0
_HOUR = 3600.0
_FAMILIES: dict[str, dict[str, tuple[float, float]]] = {
    "water flux (kg m-2 s-1)": {
        "kg m-2 s-1": (1.0, 0.0),
        "mm s-1": (1.0, 0.0),
        "mm h-1": (1.0 / _HOUR, 0.0),
        "mm day-1": (1.0 / _DAY, 0.0),
        "kg m-2 h-1": (1.0 / _HOUR, 0.0),
        "kg m-2 day-1": (1.0 / _DAY, 0.0),
        "m s-1": (1000.0, 0.0),
        "m h-1": (1000.0 / _HOUR, 0.0),
        "m day-1": (1000.0 / _DAY, 0.0),
    },
    "water amount (kg m-2)": {
        "kg m-2": (1.0, 0.0),
        "mm": (1.0, 0.0),
        "cm": (10.0, 0.0),
        "m": (1000.0, 0.0),
    },
    "temperature (K)": {"K": (1.0, 0.0), "degC": (1.0, 273.15)},
    "pressure (Pa)": {"Pa": (1.0, 0.0), "hPa": (100.0, 0.0), "kPa": (1000.0, 0.0)},
    "energy flux (W m-2)": {"W m-2": (1.0, 0.0)},
    "discharge (m3 s-1)": {"m3 s-1": (1.0, 0.0)},
    "fraction (1)": {"1": (1.0, 0.0), "%": (0.01, 0.0)},
}
_FAMILY_OF: dict[str, str] = {
    unit: family for family, units in _FAMILIES.items() for unit in units
}


def normalize_units(units: str) -> str:
    """Return the canonical spelling of ``units``.

    Products use spaces, exponents follow the symbol (``m-2``), every term
    after a ``/`` is inverted, positive powers precede negative ones (each
    ordered mass/other, length, time), and known
    symbol spellings are unified (``d``/``days`` → ``day``, ``hr`` → ``h``,
    ``mbar`` → ``hPa``, ``deg_C`` → ``degC``, ``-``/``fraction`` → ``1``).
    Unrecognized spellings are returned with whitespace collapsed.
    """

    if not isinstance(units, str):
        raise TypeError("units must be a string")
    text = " ".join(units.strip().split())
    alias = _ALIASES.get(text.lower())
    if alias is not None:
        return alias
    parts = text.replace("**", "^").split("/")
    terms: list[tuple[str, int]] = []
    for index, part in enumerate(parts):
        tokens = [token for token in re.split(r"[\s*·.]+", part) if token]
        if not tokens and index:
            return text
        for token in tokens:
            match = _TERM.match(token)
            if match is None:
                return text
            symbol, power = match.groups()
            symbol = _SYMBOLS.get(symbol.lower(), symbol)
            exponent = 1 if power is None else int(power)
            if index:
                exponent = -exponent
            terms.append((symbol, exponent))
    combined: dict[str, int] = {}
    for symbol, exponent in terms:
        combined[symbol] = combined.get(symbol, 0) + exponent
    # Within each sign, mass and other symbols precede length, then time
    # (``kg s-1 m-2`` is ``kg m-2 s-1``); ties keep their written order.
    ordered = sorted(
        (item for item in combined.items() if item[1] > 0),
        key=lambda item: _ORDER.get(item[0], 0),
    ) + sorted(
        (item for item in combined.items() if item[1] < 0),
        key=lambda item: _ORDER.get(item[0], 0),
    )
    if not ordered:
        return "1"
    return " ".join(
        symbol if exponent == 1 else f"{symbol}{exponent}"
        for symbol, exponent in ordered
    )


def units_equal(left: str, right: str) -> bool:
    """Whether two spellings normalize to the same unit."""

    return normalize_units(left) == normalize_units(right)


def check_units(
    source_units: str, target_units: str, factor: float | None = None
) -> tuple[float, float]:
    """Return ``(scale, offset)`` with ``target = source * scale + offset``.

    Equal normalized spellings give ``(1.0, 0.0)``. Otherwise both units must
    belong to one family of the explicit conversion table (water flux
    ``kg m-2 s-1``/``mm s-1``/``mm h-1``/``mm day-1``/``m s-1``; water
    amount ``kg m-2``/``mm``/``m``; ``K``/``degC``; ``Pa``/``hPa``/``kPa``;
    ``W m-2``; ``m3 s-1``; ``1``/``%``). An explicit ``factor`` is the scale
    of a pair the table does not know; for a known pair it must agree with
    the table. Any other pair raises ``ValueError``.
    """

    if factor is not None:
        factor = float(factor)
        if not math.isfinite(factor) or factor <= 0:
            raise ValueError(f"unit factor must be positive and finite, got {factor}")
    source = normalize_units(source_units)
    target = normalize_units(target_units)
    family = _FAMILY_OF.get(source)
    if source == target:
        conversion = (1.0, 0.0)
    elif family is not None and _FAMILY_OF.get(target) == family:
        source_scale, source_offset = _FAMILIES[family][source]
        target_scale, target_offset = _FAMILIES[family][target]
        conversion = (
            source_scale / target_scale,
            (source_offset - target_offset) / target_scale,
        )
    elif factor is not None:
        return factor, 0.0
    else:
        raise ValueError(
            f"no known conversion from {source_units!r} ({source}) to "
            f"{target_units!r} ({target}); pass an explicit factor"
        )
    if factor is not None and (
        conversion[1] != 0.0 or not math.isclose(factor, conversion[0], rel_tol=1e-9)
    ):
        raise ValueError(
            f"explicit unit factor {factor!r} disagrees with the known conversion "
            f"from {source!r} to {target!r} (scale {conversion[0]!r}, offset "
            f"{conversion[1]!r})"
        )
    return conversion


__all__ = ["check_units", "normalize_units", "units_equal"]

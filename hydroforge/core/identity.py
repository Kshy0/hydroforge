"""Canonical host-value identities shared by caches and rank agreement.

One encoding decides when two host values are "the same" everywhere
HydroForge compares them: compiled-program cache keys and recording
fingerprints on one rank, and declaration or invocation signatures across
ranks.  ``digest63`` is the one hash folded from such an identity.
"""

from __future__ import annotations

import pickle
from collections.abc import Callable, Mapping
from datetime import datetime, timedelta
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import Any

import cftime
import numpy as np
import torch

from hydroforge.core.time import canonical_calendar
from hydroforge.core.validation import HydroForgeModel

_DIGEST_MASK = (1 << 63) - 1


def _qualified_name(kind: type[Any]) -> str:
    return f"{kind.__module__}.{kind.__qualname__}"


def _array_identity(value: np.ndarray) -> tuple[Any, ...]:
    canonical = np.ascontiguousarray(value)
    return (
        "numpy",
        canonical.dtype.str,
        tuple(value.shape),
        sha256(canonical.view(np.uint8).tobytes()).hexdigest(),
    )


def tensor_content(value: torch.Tensor) -> tuple[Any, ...]:
    """Identify a tensor by dtype, shape and exact content."""

    canonical = value.detach().to(device="cpu").contiguous().reshape(-1)
    payload = canonical.view(torch.uint8).numpy().tobytes()
    return ("torch", str(value.dtype), tuple(value.shape), sha256(payload).hexdigest())


def canonical(
    value: Any,
    *,
    tensor: Callable[[torch.Tensor], Any] | None = None,
    other: Callable[[Any], Any] | None = None,
) -> Any:
    """Encode ``value`` as equality-safe, picklable primitives.

    Scalars keep their exact type, so ``True``, ``1`` and ``1.0`` stay
    distinct, and floats compare by bit pattern.  ``tensor`` encodes tensors
    (by identity for a recording, by content for rank agreement) and
    ``other`` any remaining value; without them such values are rejected.
    """

    def encode(item: Any) -> Any:
        if item is None:
            return None
        kind = type(item)
        if kind in (bool, int, str):
            return (kind.__name__, item)
        if kind is float:
            return ("float", item.hex())
        if isinstance(item, Enum):
            return ("enum", _qualified_name(kind), encode(item.value))
        if isinstance(item, (datetime, cftime.datetime)):
            return (
                "date",
                _qualified_name(kind),
                canonical_calendar(getattr(item, "calendar", "standard")),
                item.year,
                item.month,
                item.day,
                item.hour,
                item.minute,
                item.second,
                item.microsecond,
                getattr(item, "fold", None),
                getattr(item, "has_year_zero", None),
            )
        if kind is timedelta:
            return ("timedelta", item.days, item.seconds, item.microseconds)
        if isinstance(item, tuple):
            return ("tuple", tuple(map(encode, item)))
        if isinstance(item, list):
            return ("list", tuple(map(encode, item)))
        if isinstance(item, torch.Tensor):
            if tensor is None:
                raise TypeError("tensors have no canonical host identity here")
            return tensor(item)
        if isinstance(item, torch.device):
            return ("device", item.type)
        if isinstance(item, torch.dtype):
            return ("dtype", str(item))
        if isinstance(item, np.ndarray):
            return _array_identity(item)
        if isinstance(item, np.generic):
            return _array_identity(np.asarray(item))
        if isinstance(item, Path):
            return ("path", str(item.absolute()))
        if isinstance(item, bytes):
            return ("bytes", item.hex())
        if isinstance(item, HydroForgeModel):
            fields = object.__getattribute__(item, "__dict__")
            return (
                "model",
                _qualified_name(kind),
                tuple((name, encode(fields[name])) for name in kind.model_fields),
            )
        if isinstance(item, Mapping):
            entries = ((encode(key), encode(entry)) for key, entry in item.items())
            return ("mapping", tuple(sorted(entries, key=lambda pair: repr(pair[0]))))
        if isinstance(item, (set, frozenset)):
            return ("set", tuple(sorted(map(encode, item), key=repr)))
        if other is None:
            raise TypeError(f"{kind.__name__} values have no canonical host identity")
        return other(item)

    return encode(value)


def digest63(identity: Any) -> int:
    """Fold a canonical identity into one non-negative int64."""

    payload = pickle.dumps(identity, protocol=5)
    return int.from_bytes(sha256(payload).digest()[:8], byteorder="big") & (
        _DIGEST_MASK
    )

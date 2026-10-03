"""Reproducible model-configuration manifest publication."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from hydroforge.io.files import atomic_write_text


def write_model_manifest(
    directory: Path,
    *,
    model: str,
    experiment_name: str,
    backend: str,
    device: str,
    precision: str,
    mixed_precision: bool | None,
    opened_modules: Sequence[str],
    options: Mapping[str, Any],
    options_specialization: Iterable[tuple[str, str, Any]],
    metal_emulation: str = "native",
) -> None:
    """Write the fully resolved structural and physical model identity."""

    payload = {
        "schema_version": 1,
        "model": model,
        "experiment_name": experiment_name,
        "backend": backend,
        "device": device,
        "precision": precision,
        "mixed_precision": mixed_precision,
        "opened_modules": list(opened_modules),
        "options": options,
        "options_specialization": [
            {"path": path, "role": role, "value": value}
            for path, role, value in options_specialization
        ],
    }
    if metal_emulation != "native":
        payload["metal_emulation"] = metal_emulation
    atomic_write_text(
        directory / "model_manifest.json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


__all__ = ["write_model_manifest"]

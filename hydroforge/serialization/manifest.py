"""Reproducible model-configuration manifest publication."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from hydroforge.serialization.files import atomic_write_text

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel


def write_model_manifest(model: AbstractModel, *, backend: str) -> None:
    """Write the fully resolved structural and physical model identity."""

    if model.rank != 0:
        return
    payload = {
        "schema_version": 1,
        "model": f"{type(model).__module__}.{type(model).__qualname__}",
        "experiment_name": model.experiment_name,
        "backend": backend,
        "device": str(model.device),
        "precision": model.precision,
        "mixed_precision": model.mixed_precision,
        "opened_modules": list(model.opened_modules),
        "options": model.options.resolved_dict(),
        "options_specialization": [
            {"path": path, "role": role, "value": value}
            for path, role, value in model.options.specialization_key()
        ],
    }
    atomic_write_text(
        model.output_full_dir / "model_manifest.json",
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


__all__ = ["write_model_manifest"]

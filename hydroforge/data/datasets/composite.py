"""Named datasets read together, and the factory opening aligned variables."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from datetime import timedelta
from pathlib import Path
from typing import Annotated, Any, Self

import numpy as np
import torch
from pydantic import Field, InstanceOf, validate_call

from hydroforge.core.arrays import UniqueIds
from hydroforge.core.validation import FrozenMapping, HydroForgeModel, frozen_dict
from hydroforge.data.datasets.base import CompositeDataset, ForcingDataset
from hydroforge.data.datasets.exported import ExportedDataset
from hydroforge.data.datasets.netcdf import NetCDFDataset
from hydroforge.data.datasets.space import PointSpace

_Name = Annotated[str, Field(min_length=1)]


class MultiVariableDataset(CompositeDataset):
    """Aligned named datasets whose chunks are read as ``{name: values}``.

    Every variable shares the timeline, chunking and cadence of the first and
    the same grid or point axis (with the same selection).  Grid composites
    are mapped as a whole with :meth:`build_local_mapping`; point composites select
    IDs with :meth:`selected`.
    """

    datasets: Annotated[
        FrozenMapping[_Name, InstanceOf[ForcingDataset]], Field(min_length=1)
    ]

    @classmethod
    def _declared_reference(cls, value: Mapping[str, Any]) -> Any:
        datasets = value.get("datasets")
        if isinstance(datasets, Mapping) and datasets:
            return next(iter(datasets.values()))
        return None

    def _children(self) -> tuple[tuple[str, ForcingDataset], ...]:
        return tuple(
            (f"variable {name!r}", dataset) for name, dataset in self.datasets.items()
        )

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(self.datasets)

    def _read_at(self, index: int) -> dict[str, Any]:
        return {
            name: dataset._read_at(index) for name, dataset in self.datasets.items()
        }

    def _read_source(self, index: int) -> dict[str, Any]:
        return {
            name: dataset._read_source(index) for name, dataset in self.datasets.items()
        }

    def _read_item(self, index: int) -> dict[str, Any]:
        return {name: dataset[index] for name, dataset in self.datasets.items()}

    def _check_forcing(
        self, chunk: Any, mapping: torch.Tensor | None, *, label: str
    ) -> None:
        expected = tuple(self.datasets)
        if not isinstance(chunk, Mapping) or set(chunk) != set(expected):
            observed = tuple(chunk) if isinstance(chunk, Mapping) else type(chunk)
            raise ValueError(
                f"multi-variable forcing keys must exactly match {expected}; "
                f"got {observed}"
            )
        for name, dataset in self.datasets.items():
            dataset._check_forcing(chunk[name], mapping, label=f"{label}.{name}")

    def _derived(self, derive: Callable[[ForcingDataset], ForcingDataset]) -> Self:
        """A composite of derived children, deriving a shared child once."""

        views: dict[int, ForcingDataset] = {}
        datasets = {}
        for name, dataset in self.datasets.items():
            if id(dataset) not in views:
                views[id(dataset)] = derive(dataset)
            datasets[name] = views[id(dataset)]
        return self._view(
            {"datasets": frozen_dict(datasets)},
            _reference=next(iter(datasets.values())),
        )

    def _mapped(self, source_indices: np.ndarray, target_ids: np.ndarray) -> Self:
        return self._derived(
            lambda dataset: dataset._mapped(source_indices, target_ids)
        )

    @validate_call(config=HydroForgeModel.model_config)
    def selected(self, target_ids: UniqueIds) -> Self:
        """Return a view of every point variable at ``target_ids`` in that order."""

        if not isinstance(self.space, PointSpace):
            raise TypeError("selected() requires point datasets")
        return self._derived(lambda dataset: dataset.selected(target_ids))


@validate_call(config=HydroForgeModel.model_config)
def open_multivariable(
    dataset_type: type[NetCDFDataset] | type[ExportedDataset],
    base_dir: Annotated[Path, Field(strict=False)],
    var_specs: Annotated[Mapping[_Name, Mapping[_Name, Any]], Field(min_length=1)],
    *,
    time_interval: timedelta = timedelta(days=1),
    **shared: Any,
) -> MultiVariableDataset:
    """Open one ``dataset_type`` per variable of ``var_specs`` as a composite.

    Variable ``name`` reads ``var_name=name`` with ``prefix=f"{name}_"`` unless
    its spec says otherwise; ``shared`` keywords apply to every variable and
    each spec overrides them.  Without a declared ``chunk_len``, later
    variables use the first variable's planned chunk length.
    """

    for name, spec in var_specs.items():
        if "var_name" in spec:
            raise ValueError(
                f"var_specs[{name!r}] must not override factory-owned var_name"
            )
    datasets = {}
    for name, spec in var_specs.items():
        dataset = dataset_type(
            **(
                {
                    "base_dir": base_dir,
                    "prefix": f"{name}_",
                    "time_interval": time_interval,
                }
                | shared
                | dict(spec)
                | {"var_name": name}
            )
        )
        datasets[name] = dataset
        if shared.get("chunk_len") is None:
            # Later variables share the first variable's automatic plan.
            shared["chunk_len"] = dataset.chunk_plan.chunk_len
    return MultiVariableDataset(datasets=datasets)

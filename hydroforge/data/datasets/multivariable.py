"""Source-independent composition of named forcing datasets."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import torch

from hydroforge.contracts.temporal import DatasetTemporalContract
from hydroforge.data.datasets.base import AbstractDataset, _close_dataset_tree
from hydroforge.data.datasets.gridded import GriddedDataset


class MultiVariableDataset(AbstractDataset):
    """One validated timeline composed from named single-variable sources."""

    _REFERENCE_CONFIGURATION = frozenset({
        "start_date", "end_date", "time_interval", "model_step",
        "out_dtype", "chunk_len", "spin_up_cycles", "spin_up_start_date",
        "spin_up_end_date", "calendar", "clip_negative", "upsampling",
        "reuse_count",
    })

    def __init__(
        self,
        datasets: Mapping[str, AbstractDataset],
    ) -> None:
        if not datasets:
            raise ValueError("datasets must contain at least one named source")
        self._datasets = dict(datasets)
        invalid = {
            name: type(dataset).__name__
            for name, dataset in self._datasets.items()
            if not isinstance(dataset, AbstractDataset)
        }
        if invalid:
            raise TypeError(
                f"multi-variable sources must be datasets: {invalid}"
            )
        self._view = MappingProxyType(self._datasets)
        reference = next(iter(self._datasets.values()))
        self.reference = reference
        self._gridded = isinstance(reference, GriddedDataset)
        self._temporal_contract = DatasetTemporalContract.combine({
            name: dataset.temporal_contract
            for name, dataset in self._datasets.items()
        })
        self._chunk_plan = reference.chunk_plan
        self._simulation_schedule = reference.simulation_schedule
        for name, dataset in tuple(self._datasets.items())[1:]:
            self._validate_child(name, dataset)
        self._local_indices = None
        self._desired_catchment_ids = None

    def __getattr__(self, name: str) -> Any:
        if name in self._REFERENCE_CONFIGURATION:
            return getattr(self.reference, name)
        raise AttributeError(name)

    def _validate_child(self, name: str, dataset: AbstractDataset) -> None:
        reference = self.reference
        if (
            dataset.simulation_schedule.cadence
            != self._simulation_schedule.cadence
        ):
            raise ValueError(
                f"variable {name!r} has a different model cadence"
            )
        if dataset.chunk_plan.chunk_len != self._chunk_plan.chunk_len:
            raise ValueError(f"variable {name!r} has a different chunk length")
        if isinstance(dataset, GriddedDataset) != self._gridded:
            raise TypeError("one multi-variable dataset cannot mix grid and point sources")
        reference_coordinates = np.asarray(reference.get_coordinates()[0])
        coordinates = np.asarray(dataset.get_coordinates()[0])
        if self._gridded:
            reference_y = np.asarray(reference.get_coordinates()[1])
            y = np.asarray(dataset.get_coordinates()[1])
            same = np.array_equal(reference_coordinates, coordinates) and np.array_equal(
                reference_y, y,
            )
        else:
            same = set(reference_coordinates.tolist()) == set(coordinates.tolist())
        if not same:
            raise ValueError(f"variable {name!r} uses a different spatial domain")

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(self._datasets)

    @property
    def datasets(self) -> Mapping[str, AbstractDataset]:
        return self._view

    def get_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        return self.reference.get_coordinates()

    @property
    def data_size(self) -> int:
        return self.reference.data_size

    def get_data(self, current_time: Any, chunk_len: int):
        return {
            name: dataset.get_data(current_time, chunk_len)
            for name, dataset in self._datasets.items()
        }

    def __getitem__(self, index: int):
        return {
            name: dataset[index] for name, dataset in self._datasets.items()
        }

    def __len__(self) -> int:
        return len(self.reference)

    def select(
        self,
        desired_ids: np.ndarray,
        *,
        mapping_file: str | None = None,
        device: torch.device | None = None,
        precision: Literal["float32", "float64"] = "float32",
    ) -> torch.Tensor | None:
        """Compile the source's one valid spatial selection strategy."""
        if self._gridded:
            if mapping_file is None:
                raise ValueError("gridded selection requires mapping_file")
            reference = self.reference
            mapping = reference.build_local_mapping(
                mapping_file=mapping_file,
                desired_catchment_ids=desired_ids,
                device=device,
                precision=precision,
            )
            for dataset in tuple(self._datasets.values())[1:]:
                dataset._set_local_selection(
                    reference._local_indices,
                    reference._desired_catchment_ids,
                )
            return mapping
        if mapping_file is not None:
            raise ValueError("catchment selection does not accept mapping_file")
        for dataset in self._datasets.values():
            dataset.build_local_mapping(desired_catchment_ids=desired_ids)
        return None

    def shard_forcing(
        self, chunk: Mapping[str, torch.Tensor],
        mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self._gridded:
            if mapping is None:
                raise ValueError("gridded forcing requires its compiled mapping")
            return {
                name: dataset.shard_forcing(
                    chunk[name],
                    mapping,
                )
                for name, dataset in self._datasets.items()
            }
        if mapping is not None:
            raise ValueError("catchment forcing does not accept a grid mapping")
        return {
            name: dataset.shard_forcing(chunk[name])
            for name, dataset in self._datasets.items()
        }

    def close(self) -> None:
        _close_dataset_tree(self, scope="multi-variable dataset resources")

    def _close_children(self) -> tuple[object, ...]:
        return tuple(self._datasets.values())

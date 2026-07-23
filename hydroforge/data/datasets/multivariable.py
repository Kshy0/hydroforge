"""Source-independent composition of named forcing datasets."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import torch

from hydroforge.data.datasets.base import AbstractDataset, _close_dataset_tree
from hydroforge.data.datasets.gridded import GriddedDataset


class MultiVariableDataset(AbstractDataset):
    """One validated timeline composed from named single-variable sources."""

    def __init__(
        self,
        datasets: Mapping[str, AbstractDataset],
        *,
        loader_strategy: Literal["combined", "parallel"] = "combined",
    ) -> None:
        if not datasets:
            raise ValueError("datasets must contain at least one named source")
        self._datasets = dict(datasets)
        self._view = MappingProxyType(self._datasets)
        self.loader_strategy = loader_strategy
        reference = next(iter(self._datasets.values()))
        self.reference = reference
        self._gridded = isinstance(reference, GriddedDataset)
        for name, dataset in tuple(self._datasets.items())[1:]:
            self._validate_child(name, dataset)
        super().__init__(
            start_date=reference.start_date,
            end_date=reference.end_date,
            time_interval=reference.time_interval,
            out_dtype=reference.out_dtype,
            chunk_len=reference.chunk_len,
            spin_up_cycles=reference.spin_up_cycles,
            spin_up_start_date=reference.spin_up_start_date,
            spin_up_end_date=reference.spin_up_end_date,
            calendar=reference.calendar,
            clip_negative=reference.clip_negative,
        )

    def _validate_child(self, name: str, dataset: AbstractDataset) -> None:
        reference = self.reference
        for attribute in (
            "start_date", "end_date", "time_interval", "chunk_len",
            "spin_up_cycles", "spin_up_start_date", "spin_up_end_date",
        ):
            if getattr(dataset, attribute) != getattr(reference, attribute):
                raise ValueError(
                    f"variable {name!r} has different {attribute}"
                )
        if len(dataset) != len(reference):
            raise ValueError(f"variable {name!r} has a different chunk count")
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
                dataset._local_indices = reference._local_indices
                dataset._desired_catchment_ids = reference._desired_catchment_ids
            return mapping
        if mapping_file is not None:
            raise ValueError("catchment selection does not accept mapping_file")
        for dataset in self._datasets.values():
            dataset.build_local_mapping(desired_catchment_ids=desired_ids)
        return None

    def shard_forcing(
        self, batch: Mapping[str, torch.Tensor],
        mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if self._gridded:
            if mapping is None:
                raise ValueError("gridded forcing requires its compiled mapping")
            return {
                name: dataset.shard_forcing(batch[name], mapping)
                for name, dataset in self._datasets.items()
            }
        if mapping is not None:
            raise ValueError("catchment forcing does not accept a grid mapping")
        return {
            name: dataset.shard_forcing(batch[name])
            for name, dataset in self._datasets.items()
        }

    def iter_loaders(
        self,
        *,
        loader_workers: int = 1,
        prefetch_factor: int = 2,
        pin_memory: bool = True,
    ):
        from torch.utils.data import DataLoader

        kwargs = {
            "batch_size": max(1, loader_workers),
            "shuffle": False,
            "num_workers": loader_workers,
            "pin_memory": pin_memory,
            "prefetch_factor": prefetch_factor if loader_workers > 0 else None,
        }
        if self.loader_strategy == "combined":
            yield from DataLoader(self, **kwargs)
            return
        loaders = {
            name: DataLoader(dataset, **kwargs)
            for name, dataset in self._datasets.items()
        }
        for batches in zip(*loaders.values(), strict=True):
            yield dict(zip(loaders, batches, strict=True))

    @property
    def total_steps(self) -> int:
        return self.reference.total_steps

    def time_iter(self):
        return self.reference.time_iter()

    def close(self) -> None:
        _close_dataset_tree(self, scope="multi-variable dataset resources")

    def _close_children(self) -> tuple[object, ...]:
        return tuple(self._datasets.values())

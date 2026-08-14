"""Source-independent composition of named forcing datasets."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any, Literal

import numpy as np
import torch

from hydroforge.contracts.temporal import DatasetTemporalContract
from hydroforge.data.datasets.base import AbstractDataset, _close_dataset_tree
from hydroforge.data.datasets.chunking import SourceChunk
from hydroforge.data.datasets.gridded import GriddedDataset


_NO_SPATIAL_SELECTION = object()


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
        request_token = self._chunk_plan[0].provenance
        for name, dataset in tuple(self._datasets.items())[1:]:
            self._validate_child(name, dataset)
            adopter = getattr(dataset, "_accept_chunk_provenance", None)
            if callable(adopter):
                adopter(request_token)
            else:
                dataset.chunk_plan._accept_provenance(request_token)
        self._local_indices = getattr(reference, "_local_indices", None)
        self._desired_catchment_ids = getattr(
            reference, "_desired_catchment_ids", None,
        )
        self._refresh_spatial_selection_snapshot()

    def __getattr__(self, name: str) -> Any:
        if name in self._REFERENCE_CONFIGURATION:
            return getattr(self.reference, name)
        raise AttributeError(name)

    def _validate_child(self, name: str, dataset: AbstractDataset) -> None:
        reference = self.reference
        if dataset.temporal_contract != reference.temporal_contract:
            raise ValueError(
                f"variable {name!r} has a different temporal contract"
            )
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
            # Point columns are consumed positionally by the composite read.
            # Comparing only sets would accept ``[A, B]`` versus ``[B, A]``
            # and silently attach the second variable's values to the wrong
            # catchment.  Require the canonical order here, just as the
            # arithmetic expression composite does.
            same = np.array_equal(reference_coordinates, coordinates)
            if not same:
                try:
                    same_members = (
                        reference_coordinates.shape == coordinates.shape
                        and np.array_equal(
                            np.sort(reference_coordinates),
                            np.sort(coordinates),
                        )
                    )
                except TypeError:
                    same_members = False
                if same_members:
                    raise ValueError(
                        f"variable {name!r} uses a different spatial "
                        "coordinate order"
                    )
        if not same:
            raise ValueError(f"variable {name!r} uses a different spatial domain")
        reference_selection = getattr(
            reference, "_local_indices", _NO_SPATIAL_SELECTION,
        )
        selection = getattr(dataset, "_local_indices", _NO_SPATIAL_SELECTION)
        if reference_selection is _NO_SPATIAL_SELECTION:
            aligned = selection is _NO_SPATIAL_SELECTION
        elif selection is _NO_SPATIAL_SELECTION:
            aligned = False
        elif reference_selection is None or selection is None:
            aligned = reference_selection is selection
        else:
            aligned = np.array_equal(reference_selection, selection)
        if not aligned:
            raise ValueError(
                f"variable {name!r} uses a different spatial selection"
            )

    @staticmethod
    def _selection_handle(dataset: AbstractDataset) -> object:
        return getattr(dataset, "_local_indices", _NO_SPATIAL_SELECTION)

    def _refresh_spatial_selection_snapshot(self) -> None:
        self._spatial_selection_snapshot = tuple(
            self._selection_handle(dataset)
            for dataset in self._datasets.values()
        )

    def _assert_spatial_selection_current(self) -> None:
        for (name, dataset), expected in zip(
            self._datasets.items(),
            self._spatial_selection_snapshot,
            strict=True,
        ):
            if self._selection_handle(dataset) is not expected:
                raise RuntimeError(
                    f"variable {name!r} spatial selection changed outside "
                    "the multi-variable dataset; select the composite again"
                )

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

    def _read_chunk(self, chunk: SourceChunk):
        self._assert_spatial_selection_current()
        return {
            name: dataset.read_chunk(chunk)
            for name, dataset in self._datasets.items()
        }

    def _accept_chunk_provenance(self, provenance: object) -> None:
        """Propagate a composite request token to every source child."""
        self._chunk_plan._accept_provenance(provenance)
        for dataset in self._datasets.values():
            adopter = getattr(dataset, "_accept_chunk_provenance", None)
            if callable(adopter):
                adopter(provenance)
            else:
                dataset.chunk_plan._accept_provenance(provenance)

    def get_chunk(self, chunk: SourceChunk):
        """Prepare the same request through every named child source."""

        self._assert_spatial_selection_current()
        self._chunk_plan.validate_chunk(chunk)
        return {
            name: dataset.get_chunk(chunk)
            for name, dataset in self._datasets.items()
        }

    def __getitem__(self, index: int):
        chunk = self._chunk_plan[index]
        return self.get_chunk(chunk)

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
            self._local_indices = reference._local_indices
            self._desired_catchment_ids = reference._desired_catchment_ids
            self._refresh_spatial_selection_snapshot()
            return mapping
        if mapping_file is not None:
            raise ValueError("catchment selection does not accept mapping_file")
        for dataset in self._datasets.values():
            dataset.build_local_mapping(desired_catchment_ids=desired_ids)
        for name, dataset in tuple(self._datasets.items())[1:]:
            self._validate_child(name, dataset)
        self._local_indices = self.reference._local_indices
        self._desired_catchment_ids = np.asarray(desired_ids)
        self._refresh_spatial_selection_snapshot()
        return None

    def shard_forcing(
        self, chunk: Mapping[str, torch.Tensor],
        mapping: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        self._assert_spatial_selection_current()
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

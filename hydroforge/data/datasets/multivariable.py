"""Source-independent composition of named forcing datasets."""

from __future__ import annotations

from collections.abc import Mapping
from contextlib import ExitStack, nullcontext
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
from pydantic import (
    Field,
    PrivateAttr,
    field_validator,
    model_validator,
)

from hydroforge.contracts.temporal import _combine_temporal_domains_trusted
from hydroforge.contracts.validation import HydroForgeModel, _immutable_dict
from hydroforge.data.datasets.base import (
    AbstractDataset,
    _batch_forcing_validation,
    _close_dataset_tree,
    _spatial_selection_refs,
)
from hydroforge.data.datasets.chunking import SourceChunk
from hydroforge.data.datasets.gridded import GriddedDataset
from hydroforge.data.numeric import (
    canonical_ids,
    exact_numeric_array_equal,
    immutable_array,
)


class _MultiVariableSelectionRequest(HydroForgeModel):
    """Validated spatial view request for one composite Dataset."""

    desired_ids: np.ndarray

    @field_validator("desired_ids")
    @classmethod
    def _validate_ids(cls, value: np.ndarray) -> np.ndarray:
        result = canonical_ids(value, label="desired_ids")
        if np.unique(result).size != result.size:
            raise ValueError("desired_ids must be unique")
        return immutable_array(result, order="C")


class _MultiVariableForcingRequest(HydroForgeModel):
    """One complete named forcing batch for a composite Dataset."""

    owner: Any = Field(exclude=True, repr=False)
    data: Any

    @model_validator(mode="after")
    def _canonicalize(self):
        if not isinstance(self.owner, MultiVariableDataset):
            raise ValueError("multi-variable forcing requires Dataset ownership")
        self.owner._refresh_spatial_selection()
        if not isinstance(self.data, Mapping):
            raise ValueError("multi-variable forcing must be a mapping")
        expected = tuple(self.owner.datasets)
        observed = tuple(self.data)
        if set(observed) != set(expected):
            raise ValueError(
                "multi-variable forcing keys must exactly match "
                f"{expected}; got {observed}"
            )
        from hydroforge.data.datasets.exported import ExportedDataset

        standard = all(
            type(dataset)._validate_forcing_shard
            in {
                GriddedDataset._validate_forcing_shard,
                ExportedDataset._validate_forcing_shard,
            }
            for dataset in self.owner.datasets.values()
        )
        with _batch_forcing_validation() if standard else nullcontext():
            canonical = {
                name: dataset._validate_forcing_shard(self.data[name])
                for name, dataset in self.owner.datasets.items()
            }
        object.__setattr__(self, "data", canonical)
        return self


def compile_variable_specs(
    var_specs: Any,
) -> tuple[tuple[str, dict[str, Any]], ...]:
    """Return one immutable, canonical factory specification sequence."""

    return _VariableSpecsDeclaration(var_specs=var_specs).compiled


class _VariableSpecsDeclaration(HydroForgeModel):
    var_specs: Any

    _compiled: tuple[tuple[str, dict[str, Any]], ...] = PrivateAttr()

    @model_validator(mode="after")
    def _compile(self):
        var_specs = self.var_specs

        if not isinstance(var_specs, Mapping):
            raise ValueError("var_specs must be a mapping")
        if not var_specs:
            raise ValueError("var_specs must contain at least one variable")
        compiled = []
        for name, specification in var_specs.items():
            if type(name) is not str or not name:
                raise ValueError("var_specs keys must be non-empty exact strings")
            if not isinstance(specification, Mapping):
                raise ValueError(f"var_specs[{name!r}] must be a mapping")
            options = dict(specification)
            invalid_keys = [key for key in options if type(key) is not str or not key]
            if invalid_keys:
                raise ValueError(
                    f"var_specs[{name!r}] keys must be non-empty exact strings"
                )
            if "var_name" in options:
                raise ValueError(
                    f"var_specs[{name!r}] must not override factory-owned var_name"
                )
            compiled.append((name, options))
        self._compiled = tuple(compiled)
        return self

    @property
    def compiled(self) -> tuple[tuple[str, dict[str, Any]], ...]:
        return self._compiled


class MultiVariableDataset(AbstractDataset):
    """One validated timeline composed from named single-variable sources."""

    datasets: Mapping[str, AbstractDataset] = Field(min_length=1)

    _reference: AbstractDataset = PrivateAttr()
    _gridded: bool = PrivateAttr()
    _selection_refs: tuple[object, ...] = PrivateAttr(default=())

    @model_validator(mode="before")
    @classmethod
    def _derive_dataset_identity(cls, value: Any) -> Any:
        if not isinstance(value, Mapping):
            return value
        datasets = value.get("datasets")
        if not isinstance(datasets, Mapping) or not datasets:
            return value
        reference = next(iter(datasets.values()))
        if not isinstance(reference, AbstractDataset):
            return value
        reference._refresh_spatial_selection()
        payload = dict(value)
        derived = reference._dataset_identity_arguments()
        for name, expected in derived.items():
            if name in payload and payload[name] != expected:
                raise ValueError(f"multi-variable {name} must match its child datasets")
            payload[name] = expected
        for name in ("local_indices", "desired_catchment_ids"):
            payload.setdefault(name, getattr(reference, name))
        return payload

    @field_validator("datasets")
    @classmethod
    def _validate_datasets(
        cls,
        datasets: Mapping[str, AbstractDataset],
    ) -> Mapping[str, AbstractDataset]:
        items = tuple(datasets.items())
        invalid_names = [
            name for name, _dataset in items if type(name) is not str or not name
        ]
        if invalid_names:
            raise ValueError(
                "multi-variable source names must be non-empty exact strings"
            )
        return _immutable_dict(items)

    def _composition_parts(self):
        reference = next(iter(self.datasets.values()))
        gridded = isinstance(reference, GriddedDataset)
        temporal_domain = _combine_temporal_domains_trusted(
            {name: dataset._temporal_domain for name, dataset in self.datasets.items()}
        )
        return (
            reference,
            gridded,
            temporal_domain,
            reference.chunk_plan,
            reference.simulation_schedule,
        )

    @model_validator(mode="after")
    def _compile_composition(self):
        references = _spatial_selection_refs(self.datasets.values())
        (
            self._reference,
            self._gridded,
            temporal_domain,
            chunk_plan,
            simulation_schedule,
        ) = self._composition_parts()
        if temporal_domain != self._temporal_domain:
            raise ValueError("multi-variable timeline must match its child datasets")
        for name, dataset in tuple(self.datasets.items())[1:]:
            self._validate_child(
                name,
                dataset,
                reference=self._reference,
                gridded=self._gridded,
                chunk_plan=chunk_plan,
                schedule=simulation_schedule,
            )
        self._inherit_spatial_selection(self._reference, validate=True)
        self._selection_refs = references
        read_pools = [
            dataset._read_handles
            for dataset in self.datasets.values()
            if hasattr(dataset, "_read_handles")
        ]
        if read_pools:
            shared_pool = read_pools[0]
            for dataset in self.datasets.values():
                if hasattr(dataset, "_read_handles"):
                    dataset._read_handles = shared_pool
        return self

    def _refresh_spatial_selection(self) -> None:
        references = _spatial_selection_refs(self.datasets.values())
        if len(references) == len(self._selection_refs) and all(
            current is previous
            for current, previous in zip(references, self._selection_refs, strict=True)
        ):
            return
        reference = self._reference
        for name, dataset in tuple(self.datasets.items())[1:]:
            self._validate_child(
                name,
                dataset,
                reference=reference,
                gridded=self._gridded,
                chunk_plan=reference.chunk_plan,
                schedule=reference.simulation_schedule,
            )
        self._inherit_spatial_selection(reference, validate=False)
        self._selection_refs = references

    @property
    def reference(self) -> AbstractDataset:
        return self._reference

    def _validate_child(
        self,
        name: str,
        dataset: AbstractDataset,
        *,
        reference: AbstractDataset,
        gridded: bool,
        chunk_plan,
        schedule,
    ) -> None:
        if dataset._temporal_domain != reference._temporal_domain:
            raise ValueError(f"variable {name!r} has a different temporal contract")
        if dataset.simulation_schedule.cadence != schedule.cadence:
            raise ValueError(f"variable {name!r} has a different model cadence")
        if dataset.chunk_plan.chunk_len != chunk_plan.chunk_len:
            raise ValueError(f"variable {name!r} has a different chunk length")
        if isinstance(dataset, GriddedDataset) != gridded:
            raise ValueError(
                "one multi-variable dataset cannot mix grid and point sources"
            )
        reference_axes = reference.get_coordinates()
        observed_axes = dataset.get_coordinates()
        axes = tuple(zip(reference_axes, observed_axes, strict=True))
        if not gridded:
            axes = axes[:1]
        if any(np.ma.isMaskedArray(axis) for pair in axes for axis in pair):
            raise ValueError("multi-variable spatial coordinates must not be masked")
        reference_coordinates = np.asarray(reference_axes[0])
        coordinates = np.asarray(observed_axes[0])
        same = all(exact_numeric_array_equal(left, right) for left, right in axes)
        if not gridded:
            # Point columns are positional: equal sets in a different order
            # must not attach another variable's values to the wrong catchment.
            if not same:
                try:
                    same_members = (
                        reference_coordinates.shape == coordinates.shape
                        and exact_numeric_array_equal(
                            np.sort(reference_coordinates),
                            np.sort(coordinates),
                        )
                    )
                except TypeError:
                    same_members = False
                if same_members:
                    raise ValueError(
                        f"variable {name!r} uses a different spatial coordinate order"
                    )
        if not same:
            raise ValueError(f"variable {name!r} uses a different spatial domain")
        reference_selection = reference.local_indices
        selection = dataset.local_indices
        aligned = (
            reference_selection is selection
            if reference_selection is None or selection is None
            else exact_numeric_array_equal(reference_selection, selection)
        )
        if not aligned:
            raise ValueError(f"variable {name!r} uses a different spatial selection")

    @property
    def variables(self) -> tuple[str, ...]:
        return tuple(self.datasets)

    def get_coordinates(self) -> tuple[np.ndarray, np.ndarray]:
        self._refresh_spatial_selection()
        return self.reference.get_coordinates()

    @property
    def data_size(self) -> int:
        self._refresh_spatial_selection()
        return self.reference.data_size

    def _read_chunk(self, chunk: SourceChunk):
        self._refresh_spatial_selection()
        return {
            name: dataset._read_chunk_trusted(
                dataset.chunk_plan._at_trusted(chunk.index),
            )
            for name, dataset in self.datasets.items()
        }

    def _get_chunk_trusted(self, chunk: SourceChunk):
        """Prepare one framework-produced request through every child."""

        self._refresh_spatial_selection()
        return {
            name: dataset._get_chunk_trusted(
                dataset.chunk_plan._at_trusted(chunk.index),
            )
            for name, dataset in self.datasets.items()
        }

    def __len__(self) -> int:
        return len(self.reference)

    def close(self) -> None:
        _close_dataset_tree(self, scope="multi-variable dataset resources")

    def _close_children(self) -> tuple[object, ...]:
        return tuple(self.datasets.values())


class GriddedMultiVariableDataset(MultiVariableDataset):
    """Aligned named grid sources sharing one directly installed mapping."""

    @model_validator(mode="after")
    def _require_gridded_sources(self):
        if not self._gridded:
            raise ValueError("GriddedMultiVariableDataset requires grid sources")
        return self

    def build_local_mapping(
        self,
        mapping_file: str | Path,
        desired_catchment_ids: np.ndarray | None = None,
        device: str | torch.device | None = None,
        precision: Literal["float32", "float64"] = "float32",
    ) -> torch.Tensor:
        reference = self.reference
        datasets = tuple(
            {id(dataset): dataset for dataset in self.datasets.values()}.values()
        )
        with ExitStack() as transactions:
            for dataset in datasets:
                transactions.enter_context(dataset._mapping_transaction())
            mapping = reference.build_local_mapping(
                mapping_file=mapping_file,
                desired_catchment_ids=desired_catchment_ids,
                device=device,
                precision=precision,
            )
            for dataset in datasets[1:]:
                dataset._install_local_selection(
                    source_indices=reference.local_indices,
                    target_ids=reference.desired_catchment_ids,
                    device=reference._mapping_device,
                    precision=precision,
                )
            object.__setattr__(self, "local_indices", reference.local_indices)
            object.__setattr__(
                self,
                "desired_catchment_ids",
                reference.desired_catchment_ids,
            )
        return mapping

    def shard_forcing(
        self,
        chunk: Mapping[str, torch.Tensor],
        local_mapping: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        request = _MultiVariableForcingRequest(owner=self, data=chunk)
        return self._shard_forcing_trusted(request.data, local_mapping)

    def _validate_forcing_shard(self, chunk: Any) -> dict[str, Any]:
        return _MultiVariableForcingRequest(owner=self, data=chunk).data

    def _shard_forcing_trusted(
        self,
        chunk: Mapping[str, Any],
        local_mapping: torch.Tensor,
    ) -> dict[str, Any]:
        return {
            name: dataset._shard_forcing(
                chunk[name],
                local_mapping,
            )
            for name, dataset in self.datasets.items()
        }


class ExportedMultiVariableDataset(MultiVariableDataset):
    """Aligned named point sources whose columns already match targets."""

    @model_validator(mode="after")
    def _require_point_sources(self):
        if self._gridded:
            raise ValueError("ExportedMultiVariableDataset requires point sources")
        return self

    def selected(
        self,
        desired_ids: np.ndarray,
    ) -> ExportedMultiVariableDataset:
        request = _MultiVariableSelectionRequest.model_validate(
            {"desired_ids": desired_ids},
        )
        return self._selected_trusted(request.desired_ids)

    def _selected_trusted(
        self,
        desired_ids: np.ndarray,
    ) -> ExportedMultiVariableDataset:
        """Select every child with one already canonical ID vector."""

        from hydroforge.data.datasets.exported import ExportedDataset

        selected = {
            name: (
                dataset._selected_trusted(
                    desired_catchment_ids=desired_ids,
                    time_shift_steps=None,
                )
                if isinstance(dataset, ExportedDataset)
                else dataset._selected_trusted(desired_ids)
            )
            for name, dataset in self.datasets.items()
        }
        reference = next(iter(selected.values()))
        return type(self)(
            datasets=selected,
            local_indices=reference.local_indices,
            desired_catchment_ids=reference.desired_catchment_ids,
        )

    def shard_forcing(
        self,
        chunk: Mapping[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        request = _MultiVariableForcingRequest(owner=self, data=chunk)
        return self._shard_forcing_trusted(request.data)

    def _validate_forcing_shard(self, chunk: Any) -> dict[str, Any]:
        return _MultiVariableForcingRequest(owner=self, data=chunk).data

    def _shard_forcing_trusted(
        self,
        chunk: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            name: dataset._shard_forcing_trusted(chunk[name])
            for name, dataset in self.datasets.items()
        }

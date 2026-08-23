"""Regular-grid forcing capability and spatial mapping operations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch
from pydantic import Field, PrivateAttr, field_validator, model_validator

from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.contracts.naming import validate_safe_path_component
from hydroforge.data.aggregation import build_cama_mapping
from hydroforge.data.datasets.base import (
    SourceDataset,
    _validated_forcing_shard,
)
from hydroforge.data.datasets.export import DatasetExporter
from hydroforge.data.mapping.table import MappingTable
from hydroforge.data.numeric import (
    canonical_float64,
    canonical_ids,
    immutable_array,
)
from hydroforge.data.distributed import is_rank_zero
from hydroforge.serialization.netcdf import DEFAULT_NETCDF_OPTIONS


logger = logging.getLogger(__name__)


class _BuildLocalMappingRequest(HydroForgeModel):
    """Validated request for installing one local spatial mapping."""

    mapping_file: str | Path
    desired_catchment_ids: np.ndarray | None = None
    device: torch.device | None = None
    precision: Literal["float32", "float64"] = "float32"

    @field_validator("device", mode="before")
    @classmethod
    def _validate_device(cls, value: Any) -> torch.device | None:
        return None if value is None else torch.device(value)

    @field_validator("desired_catchment_ids")
    @classmethod
    def _validate_target_ids(
        cls,
        value: np.ndarray | None,
    ) -> np.ndarray | None:
        if value is None:
            return None
        target_ids = canonical_ids(
            value,
            label="desired_catchment_ids",
        )
        if np.unique(target_ids).size != target_ids.size:
            raise ValueError("desired_catchment_ids must be unique")
        return immutable_array(target_ids, order="C")


class _GenerateMappingTableRequest(HydroForgeModel):
    """Complete public declaration for offline Dataset mapping generation."""

    map_dir: str | Path
    out_dir: str | Path
    npz_file: str = "grid_mapping.npz"
    mapinfo_txt: str = "location.txt"
    hires_tag: str | None = "1min"
    lowres_idx_precision: str = "<i4"
    hires_idx_precision: str = "<i2"
    map_precision: str = "<f4"
    parameter_nc: str | Path | None = None
    allow_oob_zero: bool = Field(default=False, strict=True)
    source_nan_policy: Literal["keep", "drop", "nearest"] = "keep"
    source_nan_mask: np.ndarray | None = None

    @model_validator(mode="after")
    def _validate_declaration(self):
        object.__setattr__(
            self,
            "npz_file",
            validate_safe_path_component(self.npz_file, label="npz_file"),
        )
        if self.source_nan_mask is not None:
            if np.ma.isMaskedArray(self.source_nan_mask):
                raise ValueError("source_nan_mask must not be a masked array")
            if self.source_nan_mask.dtype != np.dtype(np.bool_):
                raise ValueError("source_nan_mask must use exact boolean dtype")
            mask = np.array(
                self.source_nan_mask,
                dtype=np.bool_,
                order="C",
                copy=True,
            )
            object.__setattr__(
                self,
                "source_nan_mask",
                immutable_array(mask, order="C"),
            )
        return self


class _MappingNanMaskRequest(HydroForgeModel):
    """Bind a caller or storage-derived NaN mask to one built mapping."""

    mapping: MappingTable
    nan_mask: np.ndarray | None
    policy: Literal["drop", "nearest"]

    @model_validator(mode="after")
    def _validate_mask(self):
        if self.nan_mask is None:
            raise ValueError(
                "dataset cannot infer a source NaN mask; pass "
                "source_nan_mask explicitly or use source_nan_policy='keep'"
            )
        if np.ma.isMaskedArray(self.nan_mask):
            raise ValueError("source_nan_mask must not be a masked array")
        if self.nan_mask.dtype != np.dtype(np.bool_):
            raise ValueError("source_nan_mask must use exact boolean dtype")
        if self.nan_mask.shape != self.mapping._source_shape:
            raise ValueError(
                f"source_nan_mask shape {self.nan_mask.shape} does not match "
                f"mapping source shape {self.mapping._source_shape}"
            )
        mask = np.array(self.nan_mask, dtype=np.bool_, order="C", copy=True)
        object.__setattr__(
            self, "nan_mask", immutable_array(mask, order="C"),
        )
        return self

    @property
    def valid_source_mask(self) -> np.ndarray:
        return np.logical_not(self.nan_mask)


def _mapping_axis_matches(
    observed: np.ndarray,
    expected: np.ndarray,
) -> bool:
    """Compare the exact numeric coordinates stored by both artifacts."""

    try:
        source = canonical_float64(
            observed,
            label="observed mapping coordinates",
        )
        reference = canonical_float64(
            expected,
            label="expected mapping coordinates",
        )
    except (TypeError, ValueError, OverflowError):
        return False
    if source.shape != reference.shape:
        return False
    return np.array_equal(source, reference)


class GriddedDataset(SourceDataset, ABC):
    """A temporal source with a regular horizontal grid and mapping support."""

    _mapping_device: torch.device | None = PrivateAttr(default=None)
    _mapping_precision: Literal["float32", "float64"] | None = PrivateAttr(
        default=None,
    )

    def _get_first_frame_nan_mask(self) -> Optional[np.ndarray]:
        """Return a flat full-grid NaN mask for mapping generation, if supported."""
        return None

    def _shard_forcing(
        self,
        chunk_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
        local_mapping: torch.Tensor,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Map grid data to catchments and handle distributed sync.

        Expected input shape:
          - (T, N) for single trial
          - (T, K, N) for K trials

        N matches the active source-grid axis installed on this Dataset.
        Output shape: (M, C) where M is the product of non-spatial dims, C = number of catchments.
        """
        if isinstance(chunk_data, dict):
            return {
                name: self._shard_forcing(block, local_mapping)
                for name, block in chunk_data.items()
            }

        if chunk_data.dim() == 2:
            flat = chunk_data
        else:
            T, K, N = chunk_data.shape
            flat = chunk_data.reshape(T * K, N)
        if self.clip_negative:
            flat = torch.clamp_min(flat, 0)

        out = (flat @ local_mapping).contiguous()

        if chunk_data.dim() == 3:
            T, K, _ = chunk_data.shape
            out = out.view(T, K, -1)

        return out

    def build_local_mapping(
        self,
        mapping_file: str | Path,
        desired_catchment_ids: Optional[np.ndarray] = None,
        device: Optional[Union[str, torch.device]] = None,
        precision: Literal["float32", "float64"] = "float32",
    ) -> torch.Tensor:
        """Load a v2 mapping, install its source selection, and materialize it."""

        request = _BuildLocalMappingRequest(
            mapping_file=mapping_file,
            desired_catchment_ids=desired_catchment_ids,
            device=device,
            precision=precision,
        )
        return self._build_local_mapping_trusted(
            mapping_file=request.mapping_file,
            desired_catchment_ids=request.desired_catchment_ids,
            device=request.device,
            precision=request.precision,
        )

    def _build_local_mapping_trusted(
        self,
        *,
        mapping_file: str | Path,
        desired_catchment_ids: np.ndarray | None,
        device: torch.device | None,
        precision: Literal["float32", "float64"],
    ) -> torch.Tensor:
        """Install and materialize an already validated mapping request."""

        mapping_path = Path(mapping_file)
        longitude, latitude = self.get_coordinates()
        mapping = MappingTable._load(mapping_path)
        if not (
            _mapping_axis_matches(longitude, mapping.source_x)
            and _mapping_axis_matches(latitude, mapping.source_y)
        ):
            raise ValueError(
                "dataset coordinates do not match the mapping source grid; "
                "regenerate the mapping for this dataset"
            )

        local = mapping._local(desired_catchment_ids)
        resolved_device = torch.device("cpu") if device is None else device
        self._install_local_selection(
            source_indices=local.source_indices,
            target_ids=local.target_ids,
            device=resolved_device,
            precision=precision,
        )
        dtype = torch.float32 if precision == "float32" else torch.float64
        return local.to_torch(device=resolved_device, dtype=dtype)

    def _install_local_selection(
        self,
        *,
        source_indices: np.ndarray,
        target_ids: np.ndarray,
        device: torch.device,
        precision: Literal["float32", "float64"],
    ) -> None:
        """Install mapping selection metadata without retaining its matrix."""

        object.__setattr__(self, "local_indices", source_indices)
        object.__setattr__(
            self,
            "desired_catchment_ids",
            target_ids,
        )
        self._mapping_device = device
        self._mapping_precision = precision
        self._validate_local_index_extent(
            self._grid_shape[0] * self._grid_shape[1],
            label="mapping source grid",
        )
        compute_bbox = getattr(self, "_compute_bbox_from_indices", None)
        if callable(compute_bbox):
            if source_indices.size:
                compute_bbox()
            else:
                self._bbox = None
                self._bbox_local_indices = None

    def shard_forcing(
        self,
        chunk_data: Any,
        local_mapping: torch.Tensor,
    ):
        """Map one forcing batch with the tensor returned by this dataset."""

        return self._shard_forcing(
            self._validate_forcing_shard(chunk_data),
            local_mapping,
        )

    def _validate_forcing_shard(self, chunk_data: Any):
        if self.local_indices is None:
            raise ValueError(
                "build_local_mapping() must be called before shard_forcing()"
            )
        dtype = (
            torch.float32
            if self._mapping_precision == "float32"
            else torch.float64
        )
        return _validated_forcing_shard(
            chunk_data,
            columns=self.data_size,
            dtype=dtype,
            device=self._mapping_device,
        )

    def export_climatology(
        self,
        out_path: str | Path,
        local_mapping: torch.Tensor,
        var_name: str,
        dtype: Literal["float32", "float64"] = "float32",
        netcdf_options: Mapping[str, Any] = DEFAULT_NETCDF_OPTIONS,
        device: str | torch.device = "cpu",
        units: str = "m3/s",
        description: str | None = None,
    ) -> Path:
        return DatasetExporter(self).export_climatology(
            out_path=out_path,
            local_mapping=local_mapping,
            var_name=var_name,
            dtype=dtype,
            netcdf_options=netcdf_options,
            device=device,
            units=units,
            description=description,
        )

    def export_catchment_data(
        self,
        out_dir: str | Path,
        local_mapping: torch.Tensor,
        var_name: str = "var",
        dtype: Literal["float32", "float64"] = "float32",
        netcdf_options: Mapping[str, Any] = DEFAULT_NETCDF_OPTIONS,
        normalized: bool = False,
        device: str | torch.device = "cpu",
        split_by_year: bool = False,
        units: str | Dict[str, str] = "m3/s",
        description: Optional[Union[str, Dict[str, str]]] = None,
        filename: Optional[Union[str, Dict[str, str]]] = None,
    ) -> Union[Path, List[Path], Dict[str, Path], Dict[str, List[Path]]]:
        return DatasetExporter(self).export_catchment_data(
            out_dir=out_dir,
            local_mapping=local_mapping,
            var_name=var_name,
            dtype=dtype,
            netcdf_options=netcdf_options,
            normalized=normalized,
            device=device,
            split_by_year=split_by_year,
            units=units,
            description=description,
            filename=filename,
        )

    def generate_mapping_table(
        self,
        map_dir: str | Path,
        out_dir: str | Path,
        npz_file: str = "grid_mapping.npz",
        mapinfo_txt: str = "location.txt",
        hires_tag: Optional[str] = "1min",
        lowres_idx_precision: str = "<i4",
        hires_idx_precision: str = "<i2",
        map_precision: str = "<f4",
        parameter_nc: str | Path | None = None,
        allow_oob_zero: bool = False,
        source_nan_policy: Literal["keep", "drop", "nearest"] = "keep",
        source_nan_mask: Optional[np.ndarray] = None,
    ) -> Path:
        """Generate the CaMa grid mapping table and save it as an npz file.

        Thin convenience wrapper: delegates the orchestration to
        :func:`hydroforge.data.aggregation.build_cama_mapping` using this dataset's
        source coordinates, then saves the resulting :class:`MappingTable`.  When
        ``parameter_nc`` is given, rows are aligned/subset to its ``catchment_id``
        order.

        ``source_nan_policy`` controls optional source-mask specialization at
        mapping-generation time:

        - ``"keep"`` leaves the mapping untouched.
        - ``"drop"`` removes source cells that are NaN/masked in the first frame,
          preserving each catchment's original row sum when possible.
        - ``"nearest"`` does the same, then repairs catchments that become empty
          by borrowing the nearest valid source cell.
        """
        request = _GenerateMappingTableRequest(
            map_dir=map_dir,
            out_dir=out_dir,
            npz_file=npz_file,
            mapinfo_txt=mapinfo_txt,
            hires_tag=hires_tag,
            lowres_idx_precision=lowres_idx_precision,
            hires_idx_precision=hires_idx_precision,
            map_precision=map_precision,
            parameter_nc=parameter_nc,
            allow_oob_zero=allow_oob_zero,
            source_nan_policy=source_nan_policy,
            source_nan_mask=source_nan_mask,
        )

        ro_lon, ro_lat = self.get_coordinates()
        mapping = build_cama_mapping(
            ro_lon,
            ro_lat,
            request.map_dir,
            hires_tag=request.hires_tag,
            mapinfo_txt=request.mapinfo_txt,
            lowres_idx_precision=request.lowres_idx_precision,
            hires_idx_precision=request.hires_idx_precision,
            map_precision=request.map_precision,
            parameter_nc=request.parameter_nc,
            allow_oob_zero=request.allow_oob_zero,
            producer=f"{type(self).__name__}.generate_mapping_table",
        )
        if request.source_nan_policy != "keep":
            nan_mask = request.source_nan_mask
            if nan_mask is None:
                nan_mask = self._get_first_frame_nan_mask()
            mask_request = _MappingNanMaskRequest(
                mapping=mapping,
                nan_mask=nan_mask,
                policy=request.source_nan_policy,
            )
            empty_row_policy = "nearest" if mask_request.policy == "nearest" else "zero"
            mapping = mapping._with_source_mask_trusted(
                mask_request.valid_source_mask,
                empty_row_policy=empty_row_policy,
                preserve_row_sum=True,
            )
            if is_rank_zero():
                logger.info(
                    "source_nan_policy=%r removed %d NaN source cells and "
                    "repaired %d empty targets",
                    mask_request.policy,
                    int(mask_request.nan_mask.sum()),
                    mapping.metadata.get("source_mask_repaired_rows", 0),
                )

        output_path = Path(request.out_dir) / request.npz_file
        mapping.save(output_path)
        logger.info(
            "Saved grid mapping to %s: shape=%s, nnz=%d, source=%dx%d",
            output_path,
            mapping.matrix.shape,
            mapping.matrix.nnz,
            len(ro_lon),
            len(ro_lat),
        )
        return output_path

    @abstractmethod
    def get_coordinates(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        To be implemented by subclasses, returns the coordinates of the dataset.
        """
        ...

    @property
    def data_size(self) -> int:
        """Return the number of source grid points loaded per timestep."""
        if self.local_indices is not None:
            return len(self.local_indices)
        # Full grid size from coordinates
        lon, lat = self.get_coordinates()
        return len(lon) * len(lat)

    @property
    def _grid_shape(self) -> Tuple[int, int]:
        """
        Returns (ny, nx) = (lat_size, lon_size) grid dimensions.

        Spatial convention: (Y, X) = (lat, lon)
        """
        lon, lat = self.get_coordinates()
        return (len(lat), len(lon))

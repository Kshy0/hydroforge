"""Regular-grid forcing capability and spatial mapping operations."""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Mapping
from functools import cached_property
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import numpy as np
import torch

from hydroforge.data.aggregation import build_cama_mapping
from hydroforge.data.datasets.base import AbstractDataset
from hydroforge.data.datasets.export import DatasetExporter
from hydroforge.data.mapping import MappingTable
from hydroforge.data.distributed import is_rank_zero
from hydroforge.serialization.netcdf import DEFAULT_NETCDF_OPTIONS


logger = logging.getLogger(__name__)


class GriddedDataset(AbstractDataset, ABC):
    """A temporal source with a regular horizontal grid and mapping support."""

    @cached_property
    def _exporter(self) -> DatasetExporter:
        return DatasetExporter(self)

    def export_climatology(
        self,
        out_path: Union[str, Path],
        local_mapping: torch.Tensor,
        var_name: str,
        dtype: Literal["float32", "float64"] = "float32",
        netcdf_options: Mapping[str, Any] = DEFAULT_NETCDF_OPTIONS,
        device: Union[str, torch.device] = "cpu",
        units: str = "m3/s",
        description: Optional[str] = None,
    ) -> Path:
        return self._exporter.export_climatology(
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
        units: Union[str, Dict[str, str]] = "m3/s",
        description: Optional[Union[str, Dict[str, str]]] = None,
        filename: Optional[Union[str, Dict[str, str]]] = None,
    ) -> Union[Path, List[Path], Dict[str, Path], Dict[str, List[Path]]]:
        return self._exporter.export_catchment_data(
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

    def _get_first_frame_nan_mask(self) -> Optional[np.ndarray]:
        """Return a flat full-grid NaN mask for mapping generation, if supported."""
        return None

    def shard_forcing(
        self,
        batch_data: Union[torch.Tensor, Dict[str, torch.Tensor]],
        local_mapping: torch.Tensor,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Map grid data to catchments and handle distributed sync.

        Expected input shape:
          - (B, T, N) for single trial
          - (B, T, K, N) for K trials

        N should match data_size (compressed source grids after build_local_mapping).
        Output shape: (M, C) where M is the product of non-spatial dims, C = number of catchments.
        """
        if isinstance(batch_data, dict):
            return {
                name: self.shard_forcing(block, local_mapping)
                for name, block in batch_data.items()
            }

        if batch_data.dim() == 3:
            B, T, N = batch_data.shape
            flat = batch_data.reshape(B * T, N)
        elif batch_data.dim() == 4:
            B, T, K, N = batch_data.shape
            flat = batch_data.reshape(B * T * K, N)
        else:
            raise ValueError(f"batch_data must be 3D or 4D, got shape {tuple(batch_data.shape)}")

        if flat.is_floating_point():
            flat = torch.where(torch.isnan(flat), torch.zeros_like(flat), flat)
        if self.clip_negative:
            flat = torch.clamp_min(flat, 0)

        out = (flat @ local_mapping).contiguous()

        # If input was 4D (B, T, K, N), reshape output to (B*T, K, C)
        # This makes it ready for step-by-step slicing in the main loop
        if batch_data.dim() == 4:
            B, T, K, N = batch_data.shape
            # out is currently (B*T*K, C)
            # Reshape to (B*T, K, C) so that out[step] gives (K, C) for all trials
            out = out.view(B * T, K, -1)

        return out

    def build_local_mapping(
        self,
        mapping_file: str,
        desired_catchment_ids: Optional[np.ndarray] = None,
        device: Optional[torch.device] = None,
        precision: Literal["float32", "float64"] = "float32",
    ) -> torch.Tensor:
        """Load, validate and target-select the canonical spatial mapping."""
        mapping_path = Path(mapping_file)
        if not mapping_path.is_file():
            raise FileNotFoundError(f"Mapping file not found: {mapping_path}")
        if precision not in {"float32", "float64"}:
            raise ValueError("precision must be 'float32' or 'float64'")

        mapping = MappingTable.load(mapping_path)
        longitude, latitude = self.get_coordinates()
        longitude = np.asarray(longitude)
        latitude = np.asarray(latitude)
        if not (
            longitude.shape == mapping.source_x.shape
            and latitude.shape == mapping.source_y.shape
            and np.allclose(longitude, mapping.source_x, rtol=1e-5, atol=1e-8)
            and np.allclose(latitude, mapping.source_y, rtol=1e-5, atol=1e-8)
        ):
            raise ValueError(
                "dataset coordinates do not match the mapping source grid; "
                "regenerate the mapping for this dataset"
            )

        local = mapping.local(desired_catchment_ids)
        self._mapping_file = mapping_path
        self._local_indices = local.source_indices
        self._desired_catchment_ids = local.target_ids
        dtype = torch.float32 if precision == "float32" else torch.float64
        return local.to_torch(device=device, dtype=dtype)

    def generate_mapping_table(
        self,
        map_dir: str,
        out_dir: str,
        npz_file: str = "grid_mapping.npz",
        mapinfo_txt: str = "location.txt",
        hires_tag: str = "1min",
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
        if source_nan_policy not in ("keep", "drop", "nearest"):
            raise ValueError("source_nan_policy must be one of: keep, drop, nearest")

        ro_lon, ro_lat = self.get_coordinates()
        mapping = build_cama_mapping(
            ro_lon,
            ro_lat,
            map_dir,
            hires_tag=hires_tag,
            mapinfo_txt=mapinfo_txt,
            lowres_idx_precision=lowres_idx_precision,
            hires_idx_precision=hires_idx_precision,
            map_precision=map_precision,
            parameter_nc=parameter_nc,
            allow_oob_zero=allow_oob_zero,
            producer=f"{type(self).__name__}.generate_mapping_table",
        )
        if source_nan_policy != "keep":
            nan_mask = source_nan_mask
            if nan_mask is None:
                nan_mask = self._get_first_frame_nan_mask()
            if nan_mask is None:
                raise ValueError(
                    f"{type(self).__name__} cannot infer a source NaN mask. "
                    "Pass source_nan_mask explicitly or use source_nan_policy='keep'."
                )
            nan_mask = np.asarray(nan_mask, dtype=bool).reshape(-1)
            if nan_mask.size != mapping.matrix.shape[1]:
                raise ValueError(
                    f"source_nan_mask size ({nan_mask.size}) != mapping source size "
                    f"({mapping.matrix.shape[1]})"
                )
            empty_row_policy = "nearest" if source_nan_policy == "nearest" else "zero"
            mapping = mapping.with_source_mask(
                ~nan_mask,
                empty_row_policy=empty_row_policy,
                preserve_row_sum=True,
            )
            if is_rank_zero():
                logger.info(
                    "source_nan_policy=%r removed %d NaN source cells and "
                    "repaired %d empty targets",
                    source_nan_policy, int(nan_mask.sum()),
                    mapping.metadata.get("source_mask_repaired_rows", 0),
                )

        output_path = Path(out_dir) / npz_file
        mapping.save(output_path)
        logger.info(
            "Saved grid mapping to %s: shape=%s, nnz=%d, source=%dx%d",
            output_path, mapping.matrix.shape, mapping.matrix.nnz,
            len(ro_lon), len(ro_lat),
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
        """
        Returns the number of source grid points to be loaded per timestep.

        Before build_local_mapping is called: returns full grid size.
        After build_local_mapping is called: returns compressed size (active grids only).
        """
        if self._local_indices is not None:
            return len(self._local_indices)
        # Full grid size from coordinates
        lon, lat = self.get_coordinates()
        return len(lon) * len(lat)

    @property
    def grid_shape(self) -> Tuple[int, int]:
        """
        Returns (ny, nx) = (lat_size, lon_size) grid dimensions.

        Spatial convention: (Y, X) = (lat, lon)
        """
        lon, lat = self.get_coordinates()
        return (len(lat), len(lon))

# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import numpy.ma as ma
import torch
from netCDF4 import Dataset

from hydroforge.data.netcdf import (
    _as_integer_array,
    _is_scalar_integer,
    _normalize_netcdf_index,
    _output_axis,
    read_netcdf_var_sliced,
)
from hydroforge.serialization.netcdf import (
    LOGICAL_DTYPE_ATTR, atomic_netcdf_dataset, decode_netcdf_logical_array,
    netcdf_dtype_encoding,
    normalize_netcdf_variable_options,
)


@dataclass(frozen=True, slots=True, eq=False)
class _NetCDFVariableSource:
    """One complete lazy-storage binding for an input variable."""

    path: str
    alignment_dim: str | None = None
    alignment_indices: np.ndarray | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.path, (str, Path)):
            raise TypeError("NetCDF variable source path must be a string or Path")
        if not str(self.path):
            raise ValueError("NetCDF variable source path cannot be empty")
        path = str(Path(self.path))
        object.__setattr__(self, "path", path)

        aligned = self.alignment_indices is not None
        if aligned != (self.alignment_dim is not None):
            raise ValueError(
                "NetCDF variable source alignment dimension and indices must "
                "be provided together"
            )
        if not aligned:
            return
        if not isinstance(self.alignment_dim, str) or not self.alignment_dim:
            raise TypeError("NetCDF alignment dimension must be a non-empty string")
        indices = np.asarray(self.alignment_indices)
        if indices.ndim != 1:
            raise ValueError("NetCDF alignment indices must be one-dimensional")
        if indices.dtype.kind not in "iu" or indices.dtype.kind == "b":
            raise TypeError("NetCDF alignment indices must contain integers")
        if indices.flags.writeable or indices.dtype != np.int64:
            indices = indices.astype(np.int64, copy=True)
            indices.setflags(write=False)
        object.__setattr__(self, "alignment_indices", indices)

    def align_loaded(self, variable: Any, value: np.ndarray) -> np.ndarray:
        """Apply this source's reference ordering to one eager read."""

        if (
            self.alignment_indices is None
            or self.alignment_dim not in variable.dimensions
        ):
            return value
        axis = variable.dimensions.index(self.alignment_dim)
        return np.take(value, self.alignment_indices, axis=axis)

    def selectors(self, variable: Any, indices: Any) -> Any:
        """Map one reference-order selection onto the physical file axis."""

        if (
            self.alignment_indices is None
            or self.alignment_dim not in variable.dimensions
        ):
            return indices
        axis = variable.dimensions.index(self.alignment_dim)
        return InputProxy._compose_alignment_indices(
            indices,
            ndim=variable.ndim,
            axis=axis,
            alignment_idx=self.alignment_indices,
        )


@dataclass(frozen=True, slots=True)
class _InputReadRequest:
    """One orthogonal selection interpreted by resident or lazy storage."""

    name: str
    selector: Any = Ellipsis

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise TypeError("input read variable name must be a non-empty string")

        raw = self.selector
        selectors = raw if isinstance(raw, tuple) else (raw,)
        if any(value is None for value in selectors):
            raise IndexError("input subset selection does not support new axes")
        for selector in selectors:
            if selector is Ellipsis or isinstance(selector, slice):
                continue
            if isinstance(selector, (bool, np.bool_)):
                raise TypeError("input subset scalar boolean selectors are invalid")
            if isinstance(selector, (int, np.integer)):
                continue
            try:
                array = np.asarray(selector)
            except (TypeError, ValueError):
                raise TypeError(
                    f"unsupported input subset selector {type(selector).__name__}"
                ) from None
            if array.ndim == 0 and array.dtype.kind in "iu":
                continue
            if array.ndim == 1 and array.size == 0:
                continue
            if array.ndim != 1 or array.dtype.kind not in "iub":
                raise IndexError(
                    "input subset sequence selectors must be one-dimensional "
                    "integer or boolean arrays"
                )

    def select_resident(self, value: Any) -> Any:
        """Apply the same per-axis orthogonal contract as NetCDF."""

        # ``InputProxy`` accepts Python scalars as resident values. Normalize
        # shape-less values so they follow the same selection contract as
        # NumPy/torch residents without changing tensor or masked-array
        # semantics.
        if not hasattr(value, "shape"):
            value = np.asarray(value)
        shape = tuple(value.shape)
        selectors = list(_normalize_netcdf_index(self.selector, len(shape)))
        sequence_indices: list[tuple[int, np.ndarray]] = []
        for axis, selector in enumerate(selectors):
            index = _as_integer_array(selector, shape[axis])
            if index is not None:
                sequence_indices.append((axis, index))
                selectors[axis] = slice(None)
            elif _is_scalar_integer(selector):
                selectors[axis] = int(selector)
            elif (
                isinstance(value, torch.Tensor)
                and isinstance(selector, slice)
                and selector.step is not None
                and selector.step < 0
            ):
                sequence_indices.append((
                    axis,
                    np.arange(*selector.indices(shape[axis]), dtype=np.int64),
                ))
                selectors[axis] = slice(None)
        selected = value[tuple(selectors)]
        for axis, index in sequence_indices:
            output_axis = _output_axis(selectors, axis)
            if isinstance(selected, torch.Tensor):
                indices = torch.as_tensor(
                    index, dtype=torch.int64, device=selected.device,
                )
                selected = torch.index_select(selected, output_axis, indices)
            elif np.ma.isMaskedArray(selected):
                selected = np.ma.take(selected, index, axis=output_axis)
            else:
                selected = np.take(selected, index, axis=output_axis)
        return selected


class InputProxy:
    """
    A proxy class for NetCDF input/output.
    Stores data in CPU memory (numpy arrays or torch tensors).
    """

    def __init__(
        self,
        data: Dict[str, Union[np.ndarray, torch.Tensor, float, int]],
        attrs: Optional[Dict[str, Any]] = None,
        dims: Optional[Dict[str, int]] = None,
        lazy: bool = True,
        visible_vars: Optional[Set[str]] = None,
        injected_vars: Optional[Set[str]] = None,
        *,
        _sources: Optional[Mapping[str, _NetCDFVariableSource]] = None,
    ):
        self.data = data
        self.attrs = attrs or {}
        self.dims = dims or {}
        self.lazy = lazy
        self._sources = dict(_sources or {})
        invalid_sources = {
            name: type(source).__name__
            for name, source in self._sources.items()
            if not isinstance(source, _NetCDFVariableSource)
        }
        if invalid_sources:
            raise TypeError(
                f"InputProxy sources must be NetCDF variable sources: "
                f"{invalid_sources}"
            )
        self.visible_vars = (
            set(data).union(self._sources)
            if visible_vars is None else set(visible_vars)
        )
        unresolved = self.visible_vars.difference(data).difference(self._sources)
        if unresolved:
            raise ValueError(
                "InputProxy visible variables have no resident or lazy source: "
                f"{sorted(unresolved)}"
            )
        self.injected_vars = injected_vars or set()

    @property
    def source_paths(self) -> tuple[Path, ...]:
        """Return distinct physical paths behind lazy variable sources."""

        return tuple(Path(path) for path in dict.fromkeys(
            source.path for source in self._sources.values()
        ))

    @property
    def file_path(self) -> str | list[str] | None:
        """Compatibility view of :attr:`source_paths`."""

        paths = tuple(str(path) for path in self.source_paths)
        if not paths:
            return None
        return paths[0] if len(paths) == 1 else list(paths)

    def copy(self) -> InputProxy:
        """Shallow-copy proxy registries and preserve the concrete proxy type."""

        return type(self)(
            data=dict(self.data),
            attrs=dict(self.attrs),
            dims=dict(self.dims),
            lazy=self.lazy,
            visible_vars=set(self.visible_vars),
            injected_vars=set(self.injected_vars),
            _sources=dict(self._sources),
        )

    @staticmethod
    def _read_var_from_ds(ds: Dataset, var_name: str, indices: Any = None) -> np.ndarray:
        var = ds.variables[var_name]
        if indices is None:
            v = read_netcdf_var_sliced(var)
        else:
            v = read_netcdf_var_sliced(var, indices)

        v = decode_netcdf_logical_array(var, v, name=var_name)

        if ma.isMaskedArray(v):
            # Fill masked values conservatively
            if np.issubdtype(v.dtype, np.floating):
                return np.asarray(v.filled(np.nan))
            else:
                return np.asarray(v.filled(-1))
        else:
            return np.asarray(v)

    @classmethod
    def from_nc(
        cls,
        file_path: Union[str, Path, List[Union[str, Path]]],
        lazy: bool = False,
        visible_vars: Optional[Union[List[str], Set[str]]] = None,
        align_on: Optional[str] = None,
        skip_fields: Optional[Union[List[str], Set[str]]] = None,
    ) -> InputProxy:
        """
        Create an InputProxy from one or multiple NetCDF files.
        If multiple files are provided, checks for naming conflicts.
        Reads variables, dimensions, and attributes into memory or sets up lazy loading.

        Args:
            file_path: Path(s) to NetCDF file(s).
            lazy: If True, data is loaded on demand.
            visible_vars: Optional list/set of variable names to include. Others are ignored.
            align_on: Variable name to use for alignment.
                      The FIRST file encountered containing this variable serves as the REFERENCE.
                      Subsequent files will be reordered to match the order of this variable in the reference file.
            skip_fields: Optional list/set of variable names to actively exclude.
                      Complements ``visible_vars``: a field is loaded only if it is in
                      ``visible_vars`` (when set) AND not in ``skip_fields``.  Useful
                      when the same NC drives multiple models and one wants to bypass
                      a validator/consumer on a specific field (e.g. CaMaFlood's
                      uniqueness check on ``inflow_catchment_id`` when the field is
                      allowed to repeat in HydroNet).
        """
        data = {}
        attrs = {}
        dims = {}
        found_vars = set()
        sources: dict[str, _NetCDFVariableSource] = {}

        # Normalize and validate the requested schema before opening files.
        requested_visible = None if visible_vars is None else set(visible_vars)
        requested_skip = set() if skip_fields is None else set(skip_fields)
        if requested_visible is not None:
            overlap = requested_visible.intersection(requested_skip)
            if overlap:
                raise ValueError(
                    "visible_vars and skip_fields must be disjoint; "
                    f"overlap={sorted(overlap)}"
                )
        if align_on is not None:
            if align_on in requested_skip:
                raise ValueError(
                    f"align_on={align_on!r} may not be listed in skip_fields"
                )
        visible_vars = requested_visible
        skip_fields = requested_skip

        # Normalize to list
        if isinstance(file_path, (str, Path)):
            file_paths = [file_path]
        else:
            file_paths = list(file_path)
        if not file_paths:
            raise ValueError("InputProxy.from_nc requires at least one file")
        invalid_paths = [
            value for value in file_paths
            if not isinstance(value, (str, Path))
        ]
        if invalid_paths:
            raise TypeError(
                "InputProxy.from_nc file paths must be strings or Path objects; "
                f"got {[type(value).__name__ for value in invalid_paths]}"
            )
        normalized_paths = [str(Path(value)) for value in file_paths]
        if len(normalized_paths) != len(set(normalized_paths)):
            raise ValueError("InputProxy.from_nc received duplicate file paths")

        reference_keys = None
        available_vars: set[str] = set()

        for fp in file_paths:
            path_str = str(fp)
            try:
                with Dataset(path_str, "r") as ds:
                    # Alignment logic
                    alignment_idx = None
                    if align_on:
                        if align_on not in ds.variables:
                            raise KeyError(
                                f"align_on variable {align_on!r} is absent "
                                f"from {path_str!r}"
                            )
                        align_variable = ds.variables[align_on]
                        raw_keys = read_netcdf_var_sliced(align_variable)
                        if ma.isMaskedArray(raw_keys) and np.any(
                            ma.getmaskarray(raw_keys)
                        ):
                            raise ValueError(
                                f"align_on variable {align_on!r} in "
                                f"{path_str!r} contains missing keys"
                            )
                        current_keys = np.asarray(raw_keys)
                        if current_keys.ndim != 1 or len(
                            align_variable.dimensions,
                        ) != 1:
                            raise ValueError(
                                f"align_on variable {align_on!r} in "
                                f"{path_str!r} must be one-dimensional"
                            )
                        if (
                            np.issubdtype(current_keys.dtype, np.inexact)
                            and not np.isfinite(current_keys).all()
                        ):
                            raise ValueError(
                                f"align_on variable {align_on!r} in "
                                f"{path_str!r} contains non-finite keys"
                            )
                        if np.unique(current_keys).size != current_keys.size:
                            raise ValueError(
                                f"align_on variable {align_on!r} in "
                                f"{path_str!r} contains duplicate keys"
                            )
                        alignment_dim = align_variable.dimensions[0]

                        if reference_keys is None:
                            # First file with the key sets the reference order
                            reference_keys = current_keys
                        else:
                            if len(current_keys) != len(reference_keys):
                                 raise ValueError(f"Alignment error: Variable '{align_on}' in '{path_str}' has different length ({len(current_keys)}) than reference ({len(reference_keys)}).")

                            # Subsequent files are aligned to the reference
                            sorter = np.argsort(current_keys)
                            sorted_keys = current_keys[sorter]

                            # Match keys strictly
                            insert_idx = np.searchsorted(sorted_keys, reference_keys)

                            # Check range and equality
                            if np.any(insert_idx >= len(current_keys)):
                                 raise ValueError(f"Alignment failed: Key variable '{align_on}' in '{path_str}' mismatches reference keys (indices out of bounds).")

                            matched = sorted_keys[insert_idx]
                            if not np.array_equal(matched, reference_keys):
                                 raise ValueError(f"Alignment failed: Key variable '{align_on}' in '{path_str}' does not strictly match reference keys.")

                            # alignment_idx maps: index in Ref -> index in Current
                            alignment_idx = np.asarray(
                                sorter[insert_idx], dtype=np.int64,
                            )
                            alignment_idx.setflags(write=False)

                    # Merge attributes
                    for attr_name in ds.ncattrs():
                        attrs[attr_name] = ds.getncattr(attr_name)

                    # Merge dimensions
                    for dim_name, dim in ds.dimensions.items():
                        previous_size = dims.get(dim_name)
                        if previous_size is not None and previous_size != dim.size:
                            raise ValueError(
                                f"Dimension {dim_name!r} changes size across "
                                f"input files: {previous_size} vs {dim.size} "
                                f"in {path_str!r}"
                            )
                        dims[dim_name] = dim.size

                    # Merge variables and check for conflicts
                    for var_name in ds.variables:
                        available_vars.add(var_name)
                        # Visibility / skip check
                        if visible_vars is not None and var_name not in visible_vars:
                            continue
                        if var_name in skip_fields:
                            continue

                        if var_name in found_vars:
                             previous = sources.get(var_name)
                             prev_file = (
                                 None if previous is None else previous.path
                             )
                             # Skip conflict check for align key (we use the first one encountered)
                             if align_on and var_name == align_on:
                                 continue

                             if prev_file != path_str:
                                 raise ValueError(f"Naming conflict: Variable '{var_name}' exists in both '{prev_file}' and '{path_str}'")

                        found_vars.add(var_name)
                        source = _NetCDFVariableSource(
                            path=path_str,
                            alignment_dim=(
                                alignment_dim
                                if alignment_idx is not None else None
                            ),
                            alignment_indices=alignment_idx,
                        )
                        sources[var_name] = source

                        if not lazy:
                            variable = ds.variables[var_name]
                            val = cls._read_var_from_ds(ds, var_name)
                            data[var_name] = source.align_loaded(variable, val)

            except (OSError, RuntimeError) as exc:
                exc.add_note(f"while loading InputProxy data from {path_str}")
                raise

        if align_on is not None and reference_keys is None:
            raise KeyError(
                f"align_on variable {align_on!r} was not found in any input file"
            )
        if requested_visible is not None:
            missing_visible = requested_visible.difference(available_vars)
            if missing_visible:
                raise KeyError(
                    "requested visible variable(s) were not found: "
                    f"{sorted(missing_visible)}"
                )
        missing_skip = requested_skip.difference(available_vars)
        if missing_skip:
            raise KeyError(
                "requested skipped variable(s) were not found: "
                f"{sorted(missing_skip)}"
            )

        return cls(
            data, attrs, dims, lazy=lazy, visible_vars=found_vars,
            _sources=sources,
        )

    def _source_for(self, key: str) -> _NetCDFVariableSource:
        try:
            return self._sources[key]
        except KeyError:
            raise RuntimeError(
                f"Cannot lazy load variable {key!r}: no NetCDF source is bound"
            ) from None

    def _read_lazy(self, request: _InputReadRequest) -> np.ndarray:
        source = self._source_for(request.name)
        target_path = source.path

        try:
            with Dataset(target_path, "r") as ds:
                if request.name not in ds.variables:
                     raise KeyError(
                         f"Variable '{request.name}' not found in {target_path}"
                     )

                variable = ds.variables[request.name]
                final_indices = source.selectors(variable, request.selector)
                return self._read_var_from_ds(
                    ds,
                    request.name,
                    indices=final_indices,
                )
        except (OSError, RuntimeError) as exc:
            exc.add_note(
                f"while lazily loading {request.name!r} from {target_path}"
            )
            raise

    @staticmethod
    def _compose_alignment_indices(
        indices: Any, *, ndim: int, axis: int, alignment_idx: np.ndarray,
    ) -> tuple[Any, ...]:
        """Map a reference-order selection onto one source NetCDF variable."""

        selectors = list(indices if isinstance(indices, tuple) else (
            () if indices is None else (indices,)
        ))
        ellipses = [
            index for index, value in enumerate(selectors)
            if value is Ellipsis
        ]
        if len(ellipses) > 1:
            raise IndexError("NetCDF selection may contain at most one ellipsis")
        if ellipses:
            position = ellipses[0]
            missing = ndim - (len(selectors) - 1)
            if missing < 0:
                raise IndexError("too many indices for aligned NetCDF variable")
            selectors[position:position + 1] = [slice(None)] * missing
        if len(selectors) > ndim:
            raise IndexError("too many indices for aligned NetCDF variable")
        selectors.extend([slice(None)] * (ndim - len(selectors)))
        if any(value is None for value in selectors):
            raise IndexError(
                "new-axis indexing is not supported by lazy aligned NetCDF reads"
            )
        selectors[axis] = alignment_idx[selectors[axis]]
        return tuple(selectors)

    def get_subset(self, key: str, indices: Any) -> Any:
        """
        Get a subset of a variable.
        If the variable is in memory, slices it.
        If lazy, reads only the requested indices from the file.
        """
        request = _InputReadRequest(key, indices)
        if key in self.data:
            return request.select_resident(self.data[key])

        if self.lazy and key in self.visible_vars:
            return self._read_lazy(request)

        raise KeyError(f"Variable '{key}' not found in InputProxy.")

    def get_var_shape(self, key: str) -> Tuple[int, ...]:
        """
        Get the shape of a variable without loading it fully if possible.
        """
        # If in memory, return shape
        if key in self.data:
            val = self.data[key]
            # Handle list/scalar or other types if necessary, though data usually is ndarray/tensor
            if hasattr(val, "shape"):
                return tuple(val.shape)
            return ()

        # If lazy, peek at file
        if self.lazy and key in self.visible_vars:
            source = self._source_for(key)
            with Dataset(source.path, "r") as ds:
                if key not in ds.variables:
                     raise KeyError(
                         f"Variable '{key}' not found in {source.path}"
                     )
                return tuple(ds.variables[key].shape)

        raise KeyError(f"Variable '{key}' not found in InputProxy.")


    def to_nc(
        self,
        file_path: Union[str, Path],
        *,
        netcdf_options: Mapping[str, Any],
    ) -> None:
        """
        Write the stored data to a NetCDF file.
        """
        create_options = normalize_netcdf_variable_options(netcdf_options)
        with atomic_netcdf_dataset(file_path) as ds:
            # Write global attributes
            ds.setncatts(self.attrs)

            # Helper to ensure dimension exists
            def _ensure_dim(name: str, size: Optional[int], unlimited: bool = False) -> None:
                if name in ds.dimensions:
                    return
                ds.createDimension(name, None if unlimited else size)

            # Helper to infer and write variable
            def _infer_and_write_var(name: str, data: Any) -> None:
                # Convert to numpy if tensor
                if isinstance(data, torch.Tensor):
                    arr = data.detach().cpu().numpy()
                else:
                    arr = np.asarray(data)

                vtype, logical_dtype = netcdf_dtype_encoding(arr.dtype)
                arr_to_write = arr.astype(vtype, copy=False)

                # Define dimensions
                if arr.ndim == 0:
                    dims = ()
                else:
                    dims = []
                    for ax, sz in enumerate(arr.shape):
                        dim_name = f"{name}_dim{ax}"
                        _ensure_dim(dim_name, sz, unlimited=False)
                        dims.append(dim_name)

                # Create variable
                var = ds.createVariable(name, vtype, dims, **create_options)
                if logical_dtype is not None:
                    var.setncattr(LOGICAL_DTYPE_ATTR, logical_dtype)
                var[:] = arr_to_write

            # Write variables
            for name in self.keys():
                val = self[name]
                _infer_and_write_var(name, val)

    @staticmethod
    def merge(
        output_path: Union[str, Path],
        rank_paths: List[Union[str, Path]],
        variable_group_mapping: Dict[str, str],
        *,
        netcdf_options: Mapping[str, Any],
    ) -> None:
        """Merge one exact set of rank-local checkpoint files.

        Every mapped variable must occur on every rank.  HydroForge contract
        attributes must agree exactly; rank zero's complete attributes are
        retained in the merged file.  A malformed rank set is rejected before
        it can masquerade as a resumable checkpoint.
        """
        create_options = normalize_netcdf_variable_options(netcdf_options)
        if not rank_paths:
            raise ValueError("InputProxy.merge requires at least one rank file")
        normalized_paths = [str(Path(path)) for path in rank_paths]
        if len(normalized_paths) != len(set(normalized_paths)):
            raise ValueError("InputProxy.merge received duplicate rank files")
        distributed_names = set(variable_group_mapping)
        unknown_groups = set(variable_group_mapping.values()).difference(
            distributed_names
        )
        if unknown_groups:
            raise ValueError(
                "InputProxy.merge variable groups must name mapped coordinate "
                f"variables: {sorted(unknown_groups)}"
            )
        offsets: Dict[str, int] = {}
        contract_attrs: dict[str, Any] | None = None
        coordinate_groups = set(variable_group_mapping.values())
        coordinate_parts: dict[str, list[np.ndarray]] = {
            name: [] for name in coordinate_groups
        }

        with atomic_netcdf_dataset(output_path, format="NETCDF4") as merged_ds:
            for r, rank_path in enumerate(rank_paths):
                if not Path(rank_path).exists():
                    raise FileNotFoundError(f"Missing file: {rank_path}")

                with Dataset(rank_path, "r") as rank_ds:
                    attrs = {
                        name: rank_ds.getncattr(name) for name in rank_ds.ncattrs()
                    }
                    rank_contract = {
                        name: value for name, value in attrs.items()
                        if name.startswith("hydroforge_")
                    }
                    if contract_attrs is None:
                        contract_attrs = rank_contract
                        merged_ds.setncatts(attrs)
                    elif rank_contract != contract_attrs:
                        raise ValueError(
                            f"Rank checkpoint {rank_path!s} has incompatible "
                            "HydroForge contract attributes"
                        )
                    rank_variables = set(rank_ds.variables)
                    missing_distributed = distributed_names.difference(
                        rank_variables
                    )
                    if missing_distributed:
                        raise ValueError(
                            f"Rank checkpoint {rank_path!s} is missing distributed "
                            f"variables: {sorted(missing_distributed)}"
                        )
                    for coordinate in sorted(coordinate_groups):
                        raw_coordinate = rank_ds.variables[coordinate][:]
                        if ma.isMaskedArray(raw_coordinate) and np.any(
                            ma.getmaskarray(raw_coordinate)
                        ):
                            raise ValueError(
                                f"Rank checkpoint {rank_path!s} coordinate "
                                f"{coordinate!r} contains missing IDs"
                            )
                        coordinate_data = np.asarray(raw_coordinate)
                        if coordinate_data.ndim != 1:
                            raise ValueError(
                                f"Rank checkpoint coordinate {coordinate!r} "
                                "must be one-dimensional"
                            )
                        if coordinate_data.dtype.kind not in "iu":
                            raise TypeError(
                                f"Rank checkpoint coordinate {coordinate!r} "
                                "must use an integer dtype"
                            )
                        coordinate_parts[coordinate].append(coordinate_data)
                    group_lengths: dict[str, int] = {}
                    for variable, group in variable_group_mapping.items():
                        shape = tuple(rank_ds.variables[variable].shape)
                        if not shape:
                            raise ValueError(
                                f"Distributed checkpoint variable {variable!r} "
                                "must have at least one dimension"
                            )
                        previous = group_lengths.setdefault(group, shape[0])
                        if previous != shape[0]:
                            raise ValueError(
                                f"Rank checkpoint {rank_path!s} has inconsistent "
                                f"lengths in coordinate group {group!r}: "
                                f"expected {previous}, {variable!r} has {shape[0]}"
                            )
                    unexpected = rank_variables.difference(distributed_names)
                    if r > 0 and unexpected:
                        raise ValueError(
                            f"Non-root checkpoint {rank_path!s} contains global "
                            f"variables: {sorted(unexpected)}"
                        )
                    for var_name, var_in in rank_ds.variables.items():
                        is_distributed = var_name in variable_group_mapping
                        raw_data = var_in[:]
                        data = np.asarray(decode_netcdf_logical_array(
                            var_in, raw_data, name=var_name,
                        ))
                        storage_dtype, logical_dtype = netcdf_dtype_encoding(
                            data.dtype,
                        )

                        # Define/create dims and variable in merged file
                        if var_name not in merged_ds.variables:
                            # Build dims
                            if data.ndim == 0:
                                dims = ()
                            else:
                                dims = []
                                for ax, sz in enumerate(data.shape):
                                    if is_distributed and ax == 0:
                                        dname = f"{var_name}_n"
                                        # Ensure dim exists
                                        if dname not in merged_ds.dimensions:
                                            merged_ds.createDimension(dname, None) # Unlimited
                                    else:
                                        dname = f"{var_name}_dim{ax}"
                                        if dname not in merged_ds.dimensions:
                                            merged_ds.createDimension(dname, sz)
                                    dims.append(dname)

                            merged_var = merged_ds.createVariable(
                                var_name, storage_dtype, tuple(dims),
                                **create_options,
                            )
                            if logical_dtype is not None:
                                merged_var.setncattr(
                                    LOGICAL_DTYPE_ATTR, logical_dtype,
                                )
                        else:
                            merged_var = merged_ds.variables[var_name]
                            merged_logical_dtype = getattr(
                                merged_var, LOGICAL_DTYPE_ATTR, None,
                            )
                            if logical_dtype != merged_logical_dtype:
                                raise TypeError(
                                    f"Rank checkpoint variable {var_name!r} "
                                    "changes logical dtype from "
                                    f"{merged_logical_dtype!r} to "
                                    f"{logical_dtype!r}"
                                )
                            if storage_dtype != merged_var.dtype:
                                raise TypeError(
                                    f"Rank checkpoint variable {var_name!r} changes "
                                    f"dtype from {merged_var.dtype} to "
                                    f"{storage_dtype}"
                                )
                            expected_tail = tuple(merged_var.shape[1:])
                            if is_distributed and data.shape[1:] != expected_tail:
                                raise ValueError(
                                    f"Rank checkpoint variable {var_name!r} changes "
                                    f"non-partition shape from {expected_tail} to "
                                    f"{data.shape[1:]}"
                                )

                        # Write/append
                        if data.ndim == 0:
                            # Only copy from rank 0 for non-distributed scalars
                            if r == 0:
                                merged_var.assignValue(
                                    data.astype(storage_dtype, copy=False),
                                )
                        else:
                            if is_distributed:
                                off = offsets.get(var_name, 0)
                                n = data.shape[0]
                                merged_var[off : off + n, ...] = data.astype(
                                    storage_dtype, copy=False,
                                )
                                offsets[var_name] = off + n
                            else:
                                # Only copy non-distributed arrays from rank 0
                                if r == 0:
                                    merged_var[:] = data.astype(
                                        storage_dtype, copy=False,
                                    )
            for coordinate, parts in coordinate_parts.items():
                combined = np.concatenate(parts)
                if np.unique(combined).size != combined.size:
                    raise ValueError(
                        f"Distributed checkpoint coordinate {coordinate!r} "
                        "contains duplicate IDs across rank files"
                    )

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def keys(self) -> Set[str]:
        return self.visible_vars.union(self.data.keys())

    def __getitem__(self, key: str) -> Any:
        if key in self.data:
            return self.data[key]

        if self.lazy and key in self.visible_vars:
            # Cache the loaded data to avoid repeated I/O
            loaded_data = self._read_lazy(_InputReadRequest(key))
            self.data[key] = loaded_data
            return loaded_data

        raise KeyError(f"Variable '{key}' not found in InputProxy.")

    def __setitem__(self, key: str, value: Any) -> None:
        is_existing = key in self.data or key in self.visible_vars
        self.data[key] = value
        # Also expose newly injected variables to lazy-mode lookups,
        # ``keys()`` listings, and ``__contains__``.
        self.visible_vars.add(key)
        if not is_existing:
            self.injected_vars.add(key)

    def __delitem__(self, key: str) -> None:
        """Drop ``key`` completely from the proxy.

        Mirrors :meth:`drop` for a single name via the ``del proxy[key]``
        syntax.  Unknown keys raise :class:`KeyError` (dict-like).
        """
        if key not in self.data and key not in self.visible_vars:
            raise KeyError(key)
        self.drop(key)

    def drop(self, *names: str) -> "InputProxy":
        """Remove one or more variables from every internal registry.

        Clears resident data, visibility and the complete lazy source binding
        for each supplied name so downstream consumers no longer see the
        field. Unknown names are rejected before any mutation, so a misspelled
        field cannot silently alter only part of a batch.

        Returns ``self`` to allow chaining with :meth:`from_nc`.

        Example
        -------
        Bypass a CaMaFlood uniqueness check on a field that a different
        model (e.g. HydroNet) allows to repeat::

            proxy = (InputProxy
                     .from_nc("interval_params.nc", lazy=False)
                     .drop("inflow_catchment_id"))
            model = CaMaFlood(input_proxy=proxy, ...)
        """
        if not names:
            raise ValueError("InputProxy.drop requires at least one variable name")
        if any(not isinstance(name, str) or not name for name in names):
            raise TypeError("InputProxy.drop names must be non-empty strings")
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(
                f"InputProxy.drop received duplicate names: {duplicates}"
            )
        known = self.keys().union(self._sources)
        missing = set(names).difference(known)
        if missing:
            raise KeyError(
                f"InputProxy.drop variable(s) not found: {sorted(missing)}"
            )
        for name in names:
            self.data.pop(name, None)
            self.visible_vars.discard(name)
            self._sources.pop(name, None)
            self.injected_vars.discard(name)
        return self

    def __contains__(self, key: str) -> bool:
        return key in self.data or (self.lazy and key in self.visible_vars)

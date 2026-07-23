# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import numpy.ma as ma
import torch
from netCDF4 import Dataset

from hydroforge.data.netcdf import read_netcdf_var_sliced
from hydroforge.serialization.netcdf import (
    atomic_netcdf_dataset, normalize_netcdf_variable_options,
)


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
        file_path: Optional[Union[str, Path, List[Union[str, Path]]]] = None,
        visible_vars: Optional[Set[str]] = None,
        file_indices: Optional[Dict[str, np.ndarray]] = None,
        file_alignment_dims: Optional[Dict[str, str]] = None,
        injected_vars: Optional[Set[str]] = None,
    ):
        self.data = data
        self.attrs = attrs or {}
        self.dims = dims or {}
        self.lazy = lazy
        self.file_path = file_path
        self.visible_vars = (
            set(data.keys()) if visible_vars is None else set(visible_vars)
        )
        self.file_map: Dict[str, str] = {}
        self.file_indices = file_indices or {}
        self.file_alignment_dims = file_alignment_dims or {}
        self.injected_vars = injected_vars or set()

    @staticmethod
    def _read_var_from_ds(ds: Dataset, var_name: str, indices: Any = None) -> np.ndarray:
        var = ds.variables[var_name]
        if indices is None:
            v = read_netcdf_var_sliced(var)
        else:
            v = read_netcdf_var_sliced(var, indices)

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
        file_map = {}
        file_indices = {}
        file_alignment_dims = {}

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
                            alignment_idx = sorter[insert_idx]
                            file_indices[path_str] = alignment_idx
                            file_alignment_dims[path_str] = alignment_dim

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
                             prev_file = file_map.get(var_name)
                             # Skip conflict check for align key (we use the first one encountered)
                             if align_on and var_name == align_on:
                                 continue

                             if prev_file != path_str:
                                 raise ValueError(f"Naming conflict: Variable '{var_name}' exists in both '{prev_file}' and '{path_str}'")

                        found_vars.add(var_name)
                        file_map[var_name] = path_str

                        if not lazy:
                            variable = ds.variables[var_name]
                            val = cls._read_var_from_ds(ds, var_name)
                            if (
                                alignment_idx is not None
                                and alignment_dim in variable.dimensions
                            ):
                                axis = variable.dimensions.index(alignment_dim)
                                val = np.take(val, alignment_idx, axis=axis)
                            data[var_name] = val

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

        stored_paths: str | list[str] = (
            normalized_paths[0] if len(normalized_paths) == 1
            else normalized_paths
        )
        instance = cls(
            data, attrs, dims, lazy=lazy, file_path=stored_paths,
            visible_vars=found_vars, file_indices=file_indices,
            file_alignment_dims=file_alignment_dims,
        )
        instance.file_map = file_map
        return instance

    def _resolve_target_path(self, key: str) -> str:
        target_path = None

        # Priority 1: Check internal file map (populated if from_nc with multiple files)
        if self.file_map and key in self.file_map:
            target_path = self.file_map[key]

        # Priority 2: If single file path is stored, use it
        elif self.file_path and isinstance(self.file_path, (str, Path)):
             target_path = self.file_path

        # Priority 3: If file_path is a list and map failed
        elif self.file_path and isinstance(self.file_path, list):
             raise RuntimeError(f"Variable '{key}' not found in file map, and multiple files provided. Cannot disambiguate source.")

        if not target_path:
             # Last resort check: if we only have one file in list?
             if isinstance(self.file_path, list) and len(self.file_path) == 1:
                 target_path = self.file_path[0]
             else:
                raise RuntimeError(f"Cannot lazy load variable '{key}': file source not mapped and file_path is ambiguous.")

        return str(target_path)

    def _load_var(self, key: str, indices: Any = None) -> np.ndarray:
        target_path = self._resolve_target_path(key)
        alignment_idx = self.file_indices.get(target_path)
        alignment_dim = self.file_alignment_dims.get(target_path)

        try:
            with Dataset(target_path, "r") as ds:
                if key not in ds.variables:
                     raise KeyError(f"Variable '{key}' not found in {target_path}")

                variable = ds.variables[key]
                final_indices = indices
                if (
                    alignment_idx is not None
                    and alignment_dim in variable.dimensions
                ):
                    axis = variable.dimensions.index(alignment_dim)
                    final_indices = self._compose_alignment_indices(
                        indices, ndim=variable.ndim, axis=axis,
                        alignment_idx=alignment_idx,
                    )

                return self._read_var_from_ds(ds, key, indices=final_indices)
        except (OSError, RuntimeError) as exc:
            exc.add_note(f"while lazily loading {key!r} from {target_path}")
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
        if key in self.data:
            return self.data[key][indices]

        if self.lazy and key in self.visible_vars:
            return self._load_var(key, indices=indices)

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
            target_path = self._resolve_target_path(key)
            with Dataset(target_path, "r") as ds:
                if key not in ds.variables:
                     raise KeyError(f"Variable '{key}' not found in {target_path}")
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

                # Handle bool
                if arr.dtype == np.bool_:
                    vtype = "u1"
                    arr_to_write = arr.astype("u1")
                else:
                    vtype = arr.dtype
                    arr_to_write = arr

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
                        data = np.asarray(var_in[:])

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

                            # Dtype handling
                            if data.dtype == np.bool_:
                                vtype = "u1"
                            else:
                                vtype = data.dtype

                            merged_var = merged_ds.createVariable(
                                var_name, vtype, tuple(dims), **create_options
                            )
                        else:
                            merged_var = merged_ds.variables[var_name]
                            storage_dtype = (
                                np.dtype("uint8")
                                if data.dtype == np.bool_ else data.dtype
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
                                if data.dtype == np.bool_:
                                    merged_var.assignValue(data.astype("u1"))
                                else:
                                    merged_var.assignValue(data)
                        else:
                            if is_distributed:
                                off = offsets.get(var_name, 0)
                                n = data.shape[0]
                                if data.dtype == np.bool_:
                                    data = data.astype("u1")
                                merged_var[off : off + n, ...] = data
                                offsets[var_name] = off + n
                            else:
                                # Only copy non-distributed arrays from rank 0
                                if r == 0:
                                    if data.dtype == np.bool_:
                                        data = data.astype("u1")
                                    merged_var[:] = data
            for coordinate, parts in coordinate_parts.items():
                combined = np.concatenate(parts)
                if np.unique(combined).size != combined.size:
                    raise ValueError(
                        f"Distributed checkpoint coordinate {coordinate!r} "
                        "contains duplicate IDs across rank files"
                    )

    def set_variable(self, name: str, value: Any, indices: Optional[Any] = None) -> None:
        """
        Set or update a variable.

        Args:
            name: Name of the variable.
            value: New value.
            indices: Optional indices to update specific elements.
                     If None, replaces the entire variable.
        """
        if indices is not None:
            # If lazy and not in memory yet, try to load it first so we can update it
            if name not in self.data and self.lazy and name in self.visible_vars:
                self.data[name] = self._load_var(name)

            if name not in self.data:
                raise KeyError(f"Variable '{name}' not found in InputProxy, cannot update indices.")

            target = self.data[name]

            # Ensure target is mutable (numpy array or torch tensor)
            if not isinstance(target, (np.ndarray, torch.Tensor)):
                 raise TypeError(f"Variable '{name}' is of type {type(target)}, which does not support indexed assignment.")

            target[indices] = value
        else:
            is_existing = name in self.data or name in self.visible_vars
            self.data[name] = value
            self.visible_vars.add(name)
            if not is_existing:
                self.injected_vars.add(name)

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
            loaded_data = self._load_var(key)
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

        Clears ``data``, ``visible_vars``, ``file_map`` and
        ``file_indices`` for each supplied name so downstream consumers
        (including lazy loaders, validators, and model construction) no
        longer see the field. Unknown names are rejected before any mutation,
        so a misspelled field cannot silently alter only part of a batch.

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
        known = self.keys().union(self.file_map)
        missing = set(names).difference(known)
        if missing:
            raise KeyError(
                f"InputProxy.drop variable(s) not found: {sorted(missing)}"
            )
        for name in names:
            self.data.pop(name, None)
            self.visible_vars.discard(name)
            self.file_map.pop(name, None)
            self.file_indices.pop(name, None)
            self.injected_vars.discard(name)
        return self

    def __contains__(self, key: str) -> bool:
        return key in self.data or (self.lazy and key in self.visible_vars)

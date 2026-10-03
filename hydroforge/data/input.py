# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Self, cast

import numpy as np
import numpy.ma as ma
import torch
from netCDF4 import Dataset
from pydantic import (
    BeforeValidator,
    Field,
    InstanceOf,
    PrivateAttr,
    field_serializer,
    model_validator,
    validate_call,
)

from hydroforge.core.arrays import immutable_array, immutable_metadata
from hydroforge.core.errors import cleanup_on_exit
from hydroforge.core.validation import FrozenMapping, HydroForgeModel, frozen_dict
from hydroforge.io.files import FileIdentity, SourceFiles
from hydroforge.io.netcdf.encoding import (
    LOGICAL_DTYPE_ATTR,
    decode_netcdf_logical_array,
)
from hydroforge.io.netcdf.options import ensure_hdf5_plugins
from hydroforge.io.netcdf.read import (
    decoded_dtype,
    decodes_strings,
    normalize_selection,
    output_axis,
    prefer_sparse_axis,
    read_variable,
)


def _name_set(values: _InputNames, *, label: str) -> frozenset[str]:
    names = frozenset(values)
    if len(names) != len(values):
        raise ValueError(f"{label} must not contain duplicate names")
    return names


def _require_input_value(value: Any) -> Any:
    """Check one value against the InputProxy value contract."""

    if np.ma.isMaskedArray(value):
        raise ValueError("InputProxy values must not be masked arrays")
    if not (
        isinstance(value, (np.ndarray, np.generic, torch.Tensor))
        or type(value) in {bool, int, float}
    ):
        raise ValueError(
            "InputProxy values must be NumPy arrays/scalars, torch tensors, "
            "or exact bool/int/float scalars",
        )
    if isinstance(value, np.ndarray) and value.dtype.hasobject:
        raise ValueError("InputProxy arrays must not use object dtype")
    return value


def _resident(value: Any) -> Any:
    """Seal one NumPy array this proxy owns; resident values are only read."""

    if isinstance(value, np.ndarray):
        value.setflags(write=False)
    return value


def _preserve_input_value(value: Any) -> Any:
    """Take ownership of one caller value without invoking union coercion."""

    value = _require_input_value(value)
    if isinstance(value, np.ndarray):
        return _resident(np.array(value, order="K", copy=True, subok=False))
    if isinstance(value, torch.Tensor):
        return value.detach().clone(memory_format=torch.preserve_format)
    return value


def _snapshot_input_value(value: Any) -> Any:
    """Return public storage detached from one resident value."""

    if isinstance(value, np.ndarray):
        return immutable_array(value, order="K")
    if isinstance(value, torch.Tensor):
        return value.detach().clone(memory_format=torch.preserve_format)
    return value


InputValue = Annotated[
    np.ndarray | np.generic | torch.Tensor | float | int | bool,
    BeforeValidator(_preserve_input_value),
]


class _ResidentInputData(Mapping[str, InputValue]):
    """Expose resident values without exposing their storage.

    Ordinary mapping access returns an independent snapshot: immutable NumPy
    storage or a detached Tensor clone.  ``InputProxy`` reads ``_values``.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, InputValue]) -> None:
        self._values = dict(values)

    def __getitem__(self, name: str) -> InputValue:
        return _snapshot_input_value(self._values[name])

    def __contains__(self, name: object) -> bool:
        return name in self._values

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


def _netcdf_attribute_equal(left: Any, right: Any) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, np.ndarray):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and np.array_equal(left, right, equal_nan=True)
        )
    if isinstance(left, (float, np.floating)):
        return bool(left == right or (np.isnan(left) and np.isnan(right)))
    return bool(left == right)


def _freeze_netcdf_attribute(value: Any) -> Any:
    """Detach mutable NetCDF attribute arrays from their source handle."""

    if isinstance(value, np.ndarray):
        return immutable_array(value, order="K")
    return value


def _read_netcdf_input_var(
    ds: Dataset,
    var_name: str,
    indices: Any = None,
) -> np.ndarray:
    """Read one variable according to HydroForge's logical NetCDF contract."""

    variable = ds.variables[var_name]
    value = read_variable(variable, normalize_selection(indices, variable.shape))
    return _decode_netcdf_input_array(variable, value, name=var_name)


def _decode_netcdf_input_array(variable: Any, value: Any, *, name: str) -> np.ndarray:
    value = decode_netcdf_logical_array(variable, value, name=name)
    if ma.isMaskedArray(value) and np.any(ma.getmaskarray(value)):
        raise ValueError(f"NetCDF input variable {name!r} contains missing values")
    return np.asarray(value)


_INPUT_FILE_LABEL = "NetCDF input file"
_InputName = Annotated[str, Field(min_length=1)]
_InputExtent = Annotated[int, Field(ge=0)]
_InputPath = Annotated[Path, Field(strict=False)]
_InputNames = list[_InputName] | set[_InputName] | frozenset[_InputName]


class NetCDFInputSource(HydroForgeModel):
    """One complete lazy-storage binding for an input variable."""

    path: Path = Field(strict=False)
    file_identity: InstanceOf[FileIdentity]
    dimensions: tuple[_InputName, ...]
    shape: tuple[_InputExtent, ...]
    dtype: str
    alignment_dim: _InputName | None = None
    alignment_indices: np.ndarray | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        object.__setattr__(self, "dtype", str(np.dtype(self.dtype)))
        if len(self.dimensions) != len(self.shape):
            raise ValueError(
                "NetCDF variable source dimensions and shape must have equal lengths"
            )
        aligned = self.alignment_indices is not None
        if aligned != (self.alignment_dim is not None):
            raise ValueError(
                "NetCDF variable source alignment dimension and indices must "
                "be provided together"
            )
        if not aligned:
            return self
        if self.dimensions.count(self.alignment_dim) != 1:
            raise ValueError(
                "alignment dimension must occur exactly once in dimensions"
            )
        axis = self.dimensions.index(self.alignment_dim)
        indices = self.alignment_indices
        if np.ma.isMaskedArray(indices):
            if np.any(np.ma.getmaskarray(indices)):
                raise ValueError("NetCDF alignment indices contain missing positions")
            indices = np.asarray(indices)
        if indices.ndim != 1:
            raise ValueError("NetCDF alignment indices must be one-dimensional")
        if indices.dtype != np.dtype(np.int64):
            raise ValueError(
                "NetCDF alignment indices must use exact int64 dtype",
            )
        if not indices.flags.c_contiguous:
            raise ValueError("NetCDF alignment indices must be C-contiguous")
        if indices.flags.writeable:
            raise ValueError("NetCDF alignment indices must be read-only")
        if np.any(indices < 0):
            raise ValueError("NetCDF alignment indices must be nonnegative")
        if indices.size != self.shape[axis]:
            raise ValueError(
                "NetCDF alignment indices must cover the aligned dimension"
            )
        if indices.size and np.any(indices >= self.shape[axis]):
            raise ValueError("NetCDF alignment indices exceed the aligned dimension")
        if np.unique(indices).size != indices.size:
            raise ValueError(
                "NetCDF alignment indices must be a permutation of the aligned dimension"
            )
        owned_indices = immutable_array(
            indices,
            dtype=np.int64,
            order="C",
        )
        object.__setattr__(self, "alignment_indices", owned_indices)
        return self

    @property
    def numpy_dtype(self) -> np.dtype:
        return np.dtype(self.dtype)

    def align_loaded(self, value: np.ndarray) -> np.ndarray:
        """Apply this source's reference ordering to one eager read."""

        if self.alignment_indices is None:
            return value
        axis = self.dimensions.index(self.alignment_dim)
        return np.take(value, self.alignment_indices, axis=axis)

    def selectors(self, indices: Any) -> Any:
        """Map one reference-order selection onto the physical file axis."""

        if self.alignment_indices is None:
            return indices
        axis = self.dimensions.index(self.alignment_dim)
        selectors = list(indices)
        selectors[axis] = self.alignment_indices[selectors[axis]]
        return tuple(selectors)


def _source_identities(
    sources: Mapping[str, NetCDFInputSource],
) -> dict[Path, FileIdentity]:
    identities: dict[Path, FileIdentity] = {}
    for source in sources.values():
        path = source.path.absolute()
        previous = identities.setdefault(path, source.file_identity)
        if previous != source.file_identity:
            raise ValueError(
                f"NetCDF input sources for {str(path)!r} have conflicting file identities"
            )
    return identities


def _read_netcdf_selection(variable: Any, selector: tuple[Any, ...], name: str) -> Any:
    """Read and decode one normalized orthogonal selection of a lazy variable."""

    selectors = list(selector)
    gathers: list[tuple[int, np.ndarray]] = []
    for axis, index in enumerate(selectors):
        # A dense selection reads its covering range once; one HDF5 read
        # per index gap is far slower than an in-memory take.
        if (
            isinstance(index, np.ndarray)
            and index.size
            and index.dtype.kind in "iu"
            and not prefer_sparse_axis(variable, axis, index)
        ):
            low = int(index.min())
            selectors[axis] = slice(low, int(index.max()) + 1)
            gathers.append((axis, index - low))
    raw = read_variable(variable, tuple(selectors))
    # Gather before decoding: values outside the selection must not be
    # validated (e.g. missing values between selected cells).
    for axis, positions in gathers:
        raw = raw.take(positions, axis=output_axis(selectors, axis))
    # A fresh read is owned by the caller; it needs no defensive copy.
    return _require_input_value(_decode_netcdf_input_array(variable, raw, name=name))


@dataclass(frozen=True, slots=True)
class _InputProxyNetCDFPlan:
    """Private canonical payload produced by NetCDF declaration validation."""

    data: Mapping[str, Any]
    attrs: Mapping[str, Any]
    dims: Mapping[str, int]
    visible_vars: frozenset[str]
    sources: Mapping[str, NetCDFInputSource]


def _compile_input_proxy_netcdf_plan(
    *,
    paths: tuple[Path, ...],
    lazy: bool,
    visible_vars: frozenset[str] | None,
    align_on: str | None,
    skip_fields: frozenset[str],
) -> _InputProxyNetCDFPlan:
    """Inspect NetCDF sources inside the Pydantic validation boundary."""

    data: dict[str, Any] = {}
    attrs: dict[str, Any] = {}
    dims: dict[str, int] = {}
    found_vars: set[str] = set()
    sources: dict[str, NetCDFInputSource] = {}
    attribute_sources: dict[str, Path] = {}
    available_vars: set[str] = set()
    reference_keys: np.ndarray | None = None
    alignment_dims: set[str] = set()
    keyless_variables: list[tuple[Path, str, tuple[str, ...]]] = []

    ensure_hdf5_plugins()
    for path in paths:
        try:
            file_identity = FileIdentity.capture(path)
            with Dataset(path, "r") as ds:
                alignment_idx: np.ndarray | None = None
                alignment_dim: str | None = None
                if align_on is not None and align_on in ds.variables:
                    align_variable = ds.variables[align_on]
                    if (
                        len(align_variable.shape) - int(decodes_strings(align_variable))
                        != 1
                    ):
                        raise ValueError(
                            f"align_on variable {align_on!r} in {str(path)!r} must be one-dimensional"
                        )
                    raw_keys = read_variable(
                        align_variable,
                        normalize_selection(None, align_variable.shape),
                    )
                    if ma.isMaskedArray(raw_keys) and np.any(ma.getmaskarray(raw_keys)):
                        raise ValueError(
                            f"align_on variable {align_on!r} in "
                            f"{str(path)!r} contains missing keys"
                        )
                    current_keys = np.asarray(raw_keys)
                    if current_keys.ndim != 1:
                        raise ValueError(
                            f"align_on variable {align_on!r} in "
                            f"{str(path)!r} must be one-dimensional"
                        )
                    if (
                        np.issubdtype(current_keys.dtype, np.inexact)
                        and not np.isfinite(current_keys).all()
                    ):
                        raise ValueError(
                            f"align_on variable {align_on!r} in "
                            f"{str(path)!r} contains non-finite keys"
                        )
                    if np.unique(current_keys).size != current_keys.size:
                        raise ValueError(
                            f"align_on variable {align_on!r} in "
                            f"{str(path)!r} contains duplicate keys"
                        )
                    alignment_dim = align_variable.dimensions[0]
                    alignment_dims.add(alignment_dim)

                    if reference_keys is None:
                        reference_keys = current_keys
                    else:
                        if current_keys.dtype != reference_keys.dtype:
                            raise ValueError(
                                f"Alignment key {align_on!r} in "
                                f"{str(path)!r} uses dtype "
                                f"{current_keys.dtype}, expected exact dtype "
                                f"{reference_keys.dtype}"
                            )
                        if len(current_keys) != len(reference_keys):
                            raise ValueError(
                                f"Alignment key {align_on!r} in "
                                f"{str(path)!r} has length "
                                f"{len(current_keys)}, expected "
                                f"{len(reference_keys)}"
                            )
                        sorter = np.argsort(current_keys)
                        sorted_keys = current_keys[sorter]
                        insert_idx = np.searchsorted(
                            sorted_keys,
                            reference_keys,
                        )
                        if np.any(insert_idx >= len(current_keys)):
                            raise ValueError(
                                f"Alignment key {align_on!r} in "
                                f"{str(path)!r} does not cover the reference "
                                "key set"
                            )
                        matched = sorted_keys[insert_idx]
                        if not np.array_equal(matched, reference_keys):
                            raise ValueError(
                                f"Alignment key {align_on!r} in "
                                f"{str(path)!r} does not exactly match the "
                                "reference key set"
                            )
                        alignment_idx = immutable_array(
                            sorter[insert_idx],
                            dtype=np.int64,
                            order="C",
                        )

                for attr_name in ds.ncattrs():
                    value = _freeze_netcdf_attribute(
                        ds.getncattr(attr_name),
                    )
                    if attr_name in attrs and not _netcdf_attribute_equal(
                        attrs[attr_name],
                        value,
                    ):
                        raise ValueError(
                            f"Global attribute {attr_name!r} changes across "
                            f"input files: {str(attribute_sources[attr_name])!r} "
                            f"and {str(path)!r}"
                        )
                    if attr_name not in attrs:
                        attrs[attr_name] = value
                        attribute_sources[attr_name] = path

                for dim_name, dim in ds.dimensions.items():
                    previous_size = dims.get(dim_name)
                    if previous_size is not None and previous_size != dim.size:
                        raise ValueError(
                            f"Dimension {dim_name!r} changes size across "
                            f"input files: {previous_size} vs {dim.size} "
                            f"in {str(path)!r}"
                        )
                    dims[dim_name] = dim.size

                for var_name in ds.variables:
                    available_vars.add(var_name)
                    if visible_vars is not None and var_name not in visible_vars:
                        continue
                    if var_name in skip_fields:
                        continue
                    if var_name in found_vars:
                        if align_on is not None and var_name == align_on:
                            continue
                        previous = sources[var_name].path
                        raise ValueError(
                            f"Variable {var_name!r} exists in both "
                            f"{str(previous)!r} and {str(path)!r}"
                        )

                    found_vars.add(var_name)
                    variable = ds.variables[var_name]
                    dimensions, shape = (
                        tuple(variable.dimensions),
                        tuple(variable.shape),
                    )
                    dtype = decoded_dtype(variable)
                    if decodes_strings(variable):
                        dimensions, shape = dimensions[:-1], shape[:-1]
                    if align_on is not None and align_on not in ds.variables:
                        keyless_variables.append((path, var_name, dimensions))
                    aligned_variable = (
                        alignment_idx is not None and alignment_dim in dimensions
                    )
                    # An anonymous axis (no coordinate variable) of the key
                    # length likely lost the alignment dimension's name.
                    if (
                        alignment_idx is not None
                        and not aligned_variable
                        and any(
                            size == alignment_idx.size and dim not in ds.variables
                            for dim, size in zip(dimensions, shape, strict=True)
                        )
                    ):
                        raise ValueError(
                            f"Variable {var_name!r} in {str(path)!r} has an "
                            f"anonymous axis of the alignment key length "
                            f"{alignment_idx.size} but not alignment dimension "
                            f"{alignment_dim!r}, so its order cannot be "
                            "aligned; name that axis after the alignment "
                            "dimension or list the variable in skip_fields"
                        )
                    logical_dtype = getattr(
                        variable,
                        LOGICAL_DTYPE_ATTR,
                        None,
                    )
                    source = NetCDFInputSource(
                        path=path,
                        file_identity=file_identity,
                        dimensions=dimensions,
                        shape=shape,
                        dtype=(
                            str(np.dtype(np.bool_))
                            if logical_dtype == "bool"
                            else str(dtype)
                        ),
                        alignment_dim=(alignment_dim if aligned_variable else None),
                        alignment_indices=(alignment_idx if aligned_variable else None),
                    )
                    sources[var_name] = source
            file_identity.verify(path, label=_INPUT_FILE_LABEL)
        except (OSError, RuntimeError) as error:
            error.add_note(f"while inspecting InputProxy data from {str(path)}")
            raise

    if align_on is not None and reference_keys is None:
        raise ValueError(
            f"align_on variable {align_on!r} was not found in any input file"
        )
    for path, var_name, dimensions in keyless_variables:
        shared = [dim for dim in dimensions if dim in alignment_dims]
        if shared:
            raise ValueError(
                f"Variable {var_name!r} in {str(path)!r} uses alignment "
                f"dimension {shared[0]!r}, but align_on variable {align_on!r} "
                "is absent from that file, so its order cannot be aligned"
            )

    if visible_vars is not None:
        missing_visible = visible_vars.difference(available_vars)
        if missing_visible:
            raise ValueError(
                "requested visible variable(s) were not found: "
                f"{sorted(missing_visible)}"
            )
    missing_skip = skip_fields.difference(available_vars)
    if missing_skip:
        raise ValueError(
            f"requested skipped variable(s) were not found: {sorted(missing_skip)}"
        )

    if not lazy:
        files = SourceFiles(_source_identities(sources), label=_INPUT_FILE_LABEL)
        with cleanup_on_exit("eager input reads", (files.close,)):
            for name, source in sources.items():
                with files.open_netcdf(source.path) as ds:
                    value = _read_netcdf_input_var(ds, name)
                    data[name] = _resident(source.align_loaded(value))

    return _InputProxyNetCDFPlan(
        data=frozen_dict(data),
        attrs=frozen_dict(attrs),
        dims=frozen_dict(dims),
        visible_vars=frozenset(found_vars),
        sources=frozen_dict(sources),
    )


def select_values(value: Any, selector: tuple[Any, ...]) -> Any:
    """Apply one normalized orthogonal selection to an in-memory value.

    ``selector`` is the output of :func:`normalize_selection`; the result
    follows the per-axis contract of NetCDF reads.
    """

    if not hasattr(value, "shape"):
        value = np.asarray(value)
    selectors = list(selector)
    sequence_indices: list[tuple[int, np.ndarray]] = []
    for axis, item in enumerate(selectors):
        if isinstance(item, np.ndarray):
            sequence_indices.append((axis, item))
            selectors[axis] = slice(None)
        elif (
            isinstance(value, torch.Tensor)
            and isinstance(item, slice)
            and item.step is not None
            and item.step < 0
        ):
            sequence_indices.append(
                (
                    axis,
                    np.arange(
                        *item.indices(value.shape[axis]),
                        dtype=np.int64,
                    ),
                )
            )
            selectors[axis] = slice(None)
    selected = value[tuple(selectors)]
    for axis, index in sequence_indices:
        axis_out = output_axis(selectors, axis)
        if isinstance(selected, torch.Tensor):
            indices = torch.as_tensor(
                index,
                dtype=torch.int64,
                device=selected.device,
            )
            selected = torch.index_select(selected, axis_out, indices)
        else:
            selected = selected.take(index, axis=axis_out)
    return selected


class InputProxy(HydroForgeModel):
    """Construction input of a model: resident values and lazy NetCDF sources.

    Public construction copies every caller value once; ``from_nc`` and the
    functional updates own their values without a second copy, and derived
    proxies share every unchanged value.  Resident NumPy values are read-only,
    and every public read returns an independent snapshot.  Lazy variables
    are read from their file on each use; ``close()`` releases the read
    handles, which later reads reopen.
    """

    data: Mapping[_InputName, InputValue]
    attrs: Mapping[_InputName, Any] = Field(default_factory=dict)
    dims: FrozenMapping[_InputName, _InputExtent] = Field(default_factory=dict)
    lazy: bool = True
    visible_vars: frozenset[_InputName]
    injected_vars: frozenset[_InputName] = Field(default_factory=frozenset)
    sources: FrozenMapping[_InputName, NetCDFInputSource] = Field(default_factory=dict)

    _files: SourceFiles | None = PrivateAttr(default=None)

    @field_serializer("data")
    def _serialize_data(
        self,
        value: Mapping[str, InputValue],
    ) -> dict[str, InputValue]:
        return {name: value[name] for name in value}

    @model_validator(mode="before")
    @classmethod
    def _resolve_visible_default(cls, values: Any) -> Any:
        if not isinstance(values, dict) or "visible_vars" in values:
            return values
        data = values.get("data", {})
        sources = values.get("sources", {})
        if not isinstance(data, Mapping) or not isinstance(sources, Mapping):
            return {**values, "visible_vars": set()}
        return {
            **values,
            "visible_vars": frozenset(data).union(sources),
        }

    @model_validator(mode="after")
    def _validate_proxy(self) -> Self:
        _source_identities(self.sources)
        visible_vars = self.visible_vars
        unresolved = visible_vars.difference(self.data).difference(
            self.sources,
        )
        if unresolved:
            raise ValueError(
                "InputProxy visible variables have no resident or lazy source: "
                f"{sorted(unresolved)}"
            )
        hidden = frozenset(self.data).difference(visible_vars)
        if hidden:
            raise ValueError(
                f"InputProxy resident variables must be visible: {sorted(hidden)}"
            )
        lazy_only = visible_vars.difference(self.data)
        if not self.lazy and lazy_only:
            raise ValueError(
                "InputProxy with lazy=False cannot expose source-only "
                f"variables: {sorted(lazy_only)}"
            )
        injected_vars = self.injected_vars
        invalid_injected = injected_vars.difference(self.data)
        if invalid_injected:
            raise ValueError(
                "InputProxy injected_vars must identify resident visible "
                f"variables: {sorted(invalid_injected)}"
            )
        object.__setattr__(self, "data", _ResidentInputData(self.data))
        object.__setattr__(
            self,
            "attrs",
            immutable_metadata(self.attrs, label="InputProxy attrs"),
        )
        return self

    @classmethod
    def _owned(
        cls,
        data: Mapping[str, Any],
        *,
        attrs: Mapping[str, Any],
        dims: Mapping[str, int],
        lazy: bool,
        visible_vars: frozenset[str],
        injected_vars: frozenset[str],
        sources: Mapping[str, NetCDFInputSource],
    ) -> Self:
        """Assemble values this proxy already owns, without copies or revalidation."""

        return cls.model_construct(
            data=_ResidentInputData(data),
            attrs=attrs,
            dims=frozen_dict(dims),
            lazy=lazy,
            visible_vars=visible_vars,
            injected_vars=injected_vars,
            sources=frozen_dict(sources),
        )

    def _resident_items(self):
        """Iterate resident storage without creating public copies."""

        return cast(_ResidentInputData, self.data)._values.items()

    @property
    def file_path(self) -> str | list[str] | None:
        """Return the source filename or filenames for downstream diagnostics."""

        paths = tuple(
            dict.fromkeys(str(source.path) for source in self.sources.values())
        )
        if not paths:
            return None
        return paths[0] if len(paths) == 1 else list(paths)

    @validate_call(config=HydroForgeModel.model_config)
    def updated(
        self,
        *,
        values: FrozenMapping[_InputName, InputValue] | None = None,
        dimensions: FrozenMapping[_InputName, _InputExtent] | None = None,
    ) -> Self:
        """Return a proxy with added or replaced values and dimension sizes.

        Only the new values are copied; every other value is shared.
        """

        values = {} if values is None else values
        data = dict(self._resident_items())
        data.update(values)
        known = self.visible_vars.union(self.sources)
        return self._owned(
            data,
            attrs=self.attrs,
            dims={**self.dims, **({} if dimensions is None else dimensions)},
            lazy=self.lazy,
            visible_vars=self.visible_vars.union(values),
            injected_vars=self.injected_vars.union(set(values).difference(known)),
            sources=self.sources,
        )

    @validate_call(config=HydroForgeModel.model_config)
    def without(self, *names: _InputName) -> Self:
        """Return a proxy without the named variables, sharing the others."""

        removed = set(names)
        if not removed:
            raise ValueError("InputProxy.without requires at least one name")
        if len(removed) != len(names):
            raise ValueError("InputProxy.without names must be unique")
        missing = removed.difference(self.visible_vars).difference(self.sources)
        if missing:
            raise ValueError(f"InputProxy variable(s) not found: {sorted(missing)}")
        return self._owned(
            {
                name: value
                for name, value in self._resident_items()
                if name not in removed
            },
            attrs=self.attrs,
            dims=self.dims,
            lazy=self.lazy,
            visible_vars=self.visible_vars.difference(removed),
            injected_vars=self.injected_vars.difference(removed),
            sources={
                name: source
                for name, source in self.sources.items()
                if name not in removed
            },
        )

    @classmethod
    @validate_call(config=HydroForgeModel.model_config)
    def from_nc(
        cls,
        file_path: _InputPath | list[_InputPath],
        lazy: bool = False,
        visible_vars: _InputNames | None = None,
        align_on: _InputName | None = None,
        skip_fields: _InputNames | None = None,
    ) -> Self:
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
        paths = tuple(
            path.absolute()
            for path in ((file_path,) if isinstance(file_path, Path) else file_path)
        )
        if not paths:
            raise ValueError("InputProxy.from_nc requires at least one file")
        if len({str(path) for path in paths}) != len(paths):
            raise ValueError("InputProxy.from_nc received duplicate file paths")
        visible = (
            None
            if visible_vars is None
            else _name_set(visible_vars, label="visible_vars")
        )
        skipped = (
            frozenset()
            if skip_fields is None
            else _name_set(skip_fields, label="skip_fields")
        )
        if visible is not None:
            overlap = visible.intersection(skipped)
            if overlap:
                raise ValueError(
                    "visible_vars and skip_fields must be disjoint; "
                    f"overlap={sorted(overlap)}"
                )
        if align_on is not None and align_on in skipped:
            raise ValueError(f"align_on={align_on!r} may not be listed in skip_fields")
        plan = _compile_input_proxy_netcdf_plan(
            paths=paths,
            lazy=lazy,
            visible_vars=visible,
            align_on=align_on,
            skip_fields=skipped,
        )
        return cls._owned(
            plan.data,
            attrs=plan.attrs,
            dims=plan.dims,
            lazy=lazy,
            visible_vars=plan.visible_vars,
            injected_vars=frozenset(),
            sources=plan.sources,
        )

    def _require(self, key: str) -> None:
        if key not in self.visible_vars:
            raise ValueError(f"InputProxy variable {key!r} does not exist")

    def _shape(self, key: str) -> tuple[int, ...]:
        data = cast(_ResidentInputData, self.data)._values
        if key in data:
            value = data[key]
            return tuple(value.shape) if hasattr(value, "shape") else ()
        return self.sources[key].shape

    def _dtype(self, key: str) -> np.dtype | torch.dtype:
        """Return logical storage dtype without loading a lazy variable."""

        data = cast(_ResidentInputData, self.data)._values
        if key in data:
            value = data[key]
            if isinstance(value, torch.Tensor):
                return value.dtype
            return np.asarray(value).dtype
        return self.sources[key].numpy_dtype

    def _read(self, key: str, selector: tuple[Any, ...]) -> np.ndarray:
        """Read one normalized selection of a lazy variable from its file."""

        source = self.sources[key]
        try:
            with self._source_files().open_netcdf(source.path) as ds:
                return _read_netcdf_selection(
                    ds.variables[key], source.selectors(selector), key
                )
        except (OSError, RuntimeError) as exc:
            exc.add_note(f"while lazily loading {key!r} from {source.path}")
            raise

    def _value(self, key: str) -> Any:
        """Return resident storage, or read a lazy variable completely."""

        data = cast(_ResidentInputData, self.data)._values
        if key in data:
            return data[key]
        return self._read(
            key, tuple(slice(None) for _ in range(len(self.sources[key].shape)))
        )

    def _subset(self, key: str, selector: tuple[Any, ...]) -> Any:
        """Apply one normalized selection to resident storage or its file."""

        data = cast(_ResidentInputData, self.data)._values
        if key in data:
            return select_values(data[key], selector)
        return self._read(key, selector)

    def get_var_shape(self, key: str) -> tuple[int, ...]:
        """Return the shape of a variable without loading it."""

        self._require(key)
        return self._shape(key)

    def get_subset(self, key: str, indices: Any) -> Any:
        """Return a snapshot of one orthogonal selection of a variable.

        Resident variables are sliced in memory; lazy variables read only the
        selection from their file.
        """

        self._require(key)
        selectors = indices if isinstance(indices, tuple) else (indices,)
        if any(item is None for item in selectors):
            raise ValueError("input subset selection does not support new axes")
        try:
            selector = normalize_selection(indices, self._shape(key))
        except (TypeError, IndexError) as error:
            raise ValueError(f"invalid input subset selection: {error}") from error
        return _snapshot_input_value(self._subset(key, selector))

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self.visible_vars:
            return default
        return _snapshot_input_value(self._value(key))

    def keys(self) -> set[str]:
        return set(self.visible_vars)

    def __getitem__(self, key: str) -> Any:
        self._require(key)
        return _snapshot_input_value(self._value(key))

    def __contains__(self, key: str) -> bool:
        return key in self.visible_vars

    def _source_files(self) -> SourceFiles:
        """The identities of every lazy source file with process-local handles."""

        if self._files is None:
            self._files = SourceFiles(
                _source_identities(self.sources), label=_INPUT_FILE_LABEL
            )
        return self._files

    def close(self) -> None:
        """Close process-local lazy read handles; later reads reopen them."""

        if self._files is not None:
            self._files.close()

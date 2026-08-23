# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, List, Optional, Self, Set, Tuple, Union, cast

import numpy as np
import numpy.ma as ma
import torch
from netCDF4 import Dataset
from pydantic import (
    BeforeValidator,
    Field,
    PrivateAttr,
    ValidationInfo,
    field_serializer,
    model_validator,
)

from hydroforge.data.netcdf import (
    _NetCDFReadHandlePool,
    _as_integer_array,
    _is_scalar_integer,
    _normalize_integer_slice,
    _normalize_netcdf_index,
    _output_axis,
    _read_netcdf_var_sliced_trusted,
    read_netcdf_var_sliced,
)
from hydroforge.serialization.netcdf import (
    LOGICAL_DTYPE_ATTR,
    _atomic_netcdf_dataset_trusted,
    _create_netcdf_variable_trusted,
    _prepare_netcdf_variable_options_trusted,
    decode_netcdf_logical_array,
    netcdf_dtype_encoding,
)
from hydroforge.contracts.validation import (
    HydroForgeModel,
    _immutable_dict,
)
from hydroforge.data.numeric import immutable_array, immutable_metadata


def _name_set(value: Any, *, label: str) -> frozenset[str]:
    if type(value) not in {list, set, frozenset}:
        raise ValueError(f"{label} must be a list or set of strings")
    items = list(value)
    if any(type(name) is not str or not name for name in items):
        raise ValueError(f"{label} entries must be non-empty exact strings")
    if len(items) != len(set(items)):
        raise ValueError(f"{label} must not contain duplicate names")
    return frozenset(items)


def _preserve_input_value(value: Any) -> Any:
    """Take ownership of resident values without invoking union coercion."""

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
    if isinstance(value, np.ndarray):
        if value.dtype.hasobject:
            raise ValueError("InputProxy arrays must not use object dtype")
        return np.array(value, order="K", copy=True, subok=False)
    if isinstance(value, torch.Tensor):
        return value.detach().clone(memory_format=torch.preserve_format)
    return value


def _clone_input_value_trusted(value: Any) -> Any:
    """Detach storage whose type and semantics were already validated."""

    if isinstance(value, np.ndarray):
        return np.array(value, order="K", copy=True, subok=False)
    if isinstance(value, torch.Tensor):
        return value.detach().clone(memory_format=torch.preserve_format)
    return value


def _snapshot_input_value(value: Any) -> Any:
    """Return public storage detached from one trusted resident value."""

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
    """Expose resident values without exposing trusted Tensor storage.

    NumPy values use an immutable backing buffer, so sharing them is safe.
    PyTorch has no read-only Tensor, therefore ordinary mapping access returns
    a snapshot while trusted HydroForge paths use the private accessor below.
    """

    __slots__ = ("_values",)

    def __init__(self, values: Mapping[str, InputValue]) -> None:
        self._values = dict(values)

    def __getitem__(self, name: str) -> InputValue:
        return _snapshot_input_value(self._values[name])

    def __iter__(self):
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)

    def _trusted_value(self, name: str) -> InputValue:
        return self._values[name]

    def _trusted_items(self):
        return self._values.items()


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
        return bool(
            left == right
            or (np.isnan(left) and np.isnan(right))
        )
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
    value = (
        read_netcdf_var_sliced(variable)
        if indices is None
        else read_netcdf_var_sliced(variable, indices)
    )
    value = decode_netcdf_logical_array(variable, value, name=var_name)
    if ma.isMaskedArray(value) and np.any(ma.getmaskarray(value)):
        raise ValueError(
            f"NetCDF input variable {var_name!r} contains missing values"
        )
    return np.asarray(value)


class _NetCDFFileIdentity(HydroForgeModel):
    """Filesystem identity captured with one validated NetCDF schema."""

    device: int = Field(ge=0, strict=True)
    inode: int = Field(ge=0, strict=True)
    size: int = Field(ge=0, strict=True)
    mtime_ns: int = Field(ge=0, strict=True)

    @classmethod
    def _capture(cls, path: Path) -> Self:
        stat = path.stat()
        return cls.model_construct(
            device=stat.st_dev,
            inode=stat.st_ino,
            size=stat.st_size,
            mtime_ns=stat.st_mtime_ns,
        )

    def _verify(self, path: Path) -> None:
        if type(self)._capture(path) != self:
            raise RuntimeError(
                f"NetCDF input file {str(path)!r} changed after validation"
            )


class NetCDFInputSource(HydroForgeModel):
    """One complete lazy-storage binding for an input variable."""

    path: Path
    file_identity: _NetCDFFileIdentity
    dimensions: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: str
    alignment_dim: str | None = None
    alignment_indices: np.ndarray | None = None

    @model_validator(mode="after")
    def _validate_source(self) -> Self:
        if len(self.dimensions) != len(self.shape):
            raise ValueError(
                "NetCDF variable source dimensions and shape must have "
                "equal lengths"
            )
        if any(type(name) is not str or not name for name in self.dimensions):
            raise ValueError(
                "NetCDF variable source dimensions must be non-empty strings"
            )
        if any(type(size) is not int or size < 0 for size in self.shape):
            raise ValueError(
                "NetCDF variable source shape must contain nonnegative ints"
            )
        aligned = self.alignment_indices is not None
        if aligned != (self.alignment_dim is not None):
            raise ValueError(
                "NetCDF variable source alignment dimension and indices must "
                "be provided together"
            )
        if not aligned:
            return self
        if not isinstance(self.alignment_dim, str) or not self.alignment_dim:
            raise ValueError("NetCDF alignment dimension must be a non-empty string")
        indices = self.alignment_indices
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
        axis = self.dimensions.index(self.alignment_dim)
        if indices.size != self.shape[axis]:
            raise ValueError(
                "NetCDF alignment indices must cover the aligned dimension"
            )
        if indices.size and np.any(indices >= self.shape[axis]):
            raise ValueError(
                "NetCDF alignment indices exceed the aligned dimension"
            )
        owned_indices = immutable_array(
            indices, dtype=np.int64, order="C",
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
        return InputProxy._compose_alignment_indices(
            indices,
            ndim=len(self.shape),
            axis=axis,
            alignment_idx=self.alignment_indices,
        )


class _NetCDFChunkRequest(HydroForgeModel):
    """Validate one newly decoded lazy NetCDF chunk at the I/O boundary."""

    name: str
    dataset: Any = Field(exclude=True)
    selector: Any = Field(exclude=True)

    _array: np.ndarray = PrivateAttr()

    @model_validator(mode="after")
    def _read_and_validate(self) -> Self:
        variable = self.dataset.variables[self.name]
        raw = _read_netcdf_var_sliced_trusted(
            variable, self.selector,
        )
        decoded = decode_netcdf_logical_array(
            variable, raw, name=self.name,
        )
        if ma.isMaskedArray(decoded) and np.any(
            ma.getmaskarray(decoded)
        ):
            raise ValueError(
                f"NetCDF input variable {self.name!r} contains missing values"
            )
        self._array = _preserve_input_value(np.asarray(decoded))
        return self

    @property
    def array(self) -> np.ndarray:
        return self._array


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

    for path in paths:
        try:
            file_identity = _NetCDFFileIdentity._capture(path)
            with Dataset(path, "r") as ds:
                alignment_idx: np.ndarray | None = None
                alignment_dim: str | None = None
                if align_on is not None and align_on in ds.variables:
                    align_variable = ds.variables[align_on]
                    raw_keys = read_netcdf_var_sliced(align_variable)
                    if ma.isMaskedArray(raw_keys) and np.any(
                        ma.getmaskarray(raw_keys)
                    ):
                        raise ValueError(
                            f"align_on variable {align_on!r} in "
                            f"{str(path)!r} contains missing keys"
                        )
                    current_keys = np.asarray(raw_keys)
                    if (
                        current_keys.ndim != 1
                        or len(align_variable.dimensions) != 1
                    ):
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
                            sorted_keys, reference_keys,
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
                            sorter[insert_idx], dtype=np.int64, order="C",
                        )

                for attr_name in ds.ncattrs():
                    value = _freeze_netcdf_attribute(
                        ds.getncattr(attr_name),
                    )
                    if attr_name in attrs and not _netcdf_attribute_equal(
                        attrs[attr_name], value,
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
                    aligned_variable = (
                        alignment_idx is not None
                        and alignment_dim in variable.dimensions
                    )
                    logical_dtype = getattr(
                        variable, LOGICAL_DTYPE_ATTR, None,
                    )
                    source = NetCDFInputSource(
                        path=path,
                        file_identity=file_identity,
                        dimensions=tuple(variable.dimensions),
                        shape=tuple(variable.shape),
                        dtype=(
                            str(np.dtype(np.bool_))
                            if logical_dtype == "bool"
                            else str(np.dtype(variable.dtype))
                        ),
                        alignment_dim=(
                            alignment_dim
                            if aligned_variable else None
                        ),
                        alignment_indices=(
                            alignment_idx if aligned_variable else None
                        ),
                    )
                    sources[var_name] = source
                    if not lazy:
                        value = _read_netcdf_input_var(ds, var_name)
                        data[var_name] = source.align_loaded(value)
            file_identity._verify(path)
        except (OSError, RuntimeError) as error:
            error.add_note(
                f"while inspecting InputProxy data from {str(path)}"
            )
            raise

    if align_on is not None and reference_keys is None:
        raise ValueError(
            f"align_on variable {align_on!r} was not found in any input file"
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
            "requested skipped variable(s) were not found: "
            f"{sorted(missing_skip)}"
        )

    return _InputProxyNetCDFPlan(
        data=_immutable_dict(data),
        attrs=_immutable_dict(attrs),
        dims=_immutable_dict(dims),
        visible_vars=frozenset(found_vars),
        sources=_immutable_dict(sources),
    )


class _InputProxyNetCDFDeclaration(HydroForgeModel):
    """Validated declaration consumed by ``InputProxy.from_nc``."""

    file_path: str | Path | list[str | Path]
    lazy: bool = False
    visible_vars: list[str] | set[str] | frozenset[str] | None = None
    align_on: str | None = None
    skip_fields: list[str] | set[str] | frozenset[str] | None = None
    _plan: _InputProxyNetCDFPlan = PrivateAttr()

    @model_validator(mode="after")
    def _validate_open(self) -> Self:
        raw_paths = (
            [self.file_path]
            if isinstance(self.file_path, (str, Path))
            else list(self.file_path)
        )
        if not raw_paths:
            raise ValueError("InputProxy.from_nc requires at least one file")
        paths = tuple(Path(path) for path in raw_paths)
        normalized = tuple(str(path) for path in paths)
        if len(normalized) != len(set(normalized)):
            raise ValueError("InputProxy.from_nc received duplicate file paths")
        visible = (
            None
            if self.visible_vars is None
            else _name_set(self.visible_vars, label="visible_vars")
        )
        skipped = (
            set()
            if self.skip_fields is None
            else _name_set(self.skip_fields, label="skip_fields")
        )
        if self.align_on is not None and not self.align_on:
            raise ValueError("align_on must be None or a non-empty exact string")
        if visible is not None:
            overlap = visible.intersection(skipped)
            if overlap:
                raise ValueError(
                    "visible_vars and skip_fields must be disjoint; "
                    f"overlap={sorted(overlap)}"
                )
        if self.align_on is not None and self.align_on in skipped:
            raise ValueError(
                f"align_on={self.align_on!r} may not be listed in skip_fields"
            )
        object.__setattr__(self, "file_path", paths)
        object.__setattr__(self, "visible_vars", visible)
        object.__setattr__(self, "skip_fields", skipped)
        self._plan = _compile_input_proxy_netcdf_plan(
            paths=paths,
            lazy=self.lazy,
            visible_vars=visible,
            align_on=self.align_on,
            skip_fields=skipped,
        )
        return self

    @property
    def plan(self) -> _InputProxyNetCDFPlan:
        return self._plan


class _InputProxyUpdate(HydroForgeModel):
    """Validated functional update accepted by :meth:`InputProxy.updated`."""

    values: Mapping[str, InputValue] = Field(default_factory=dict)
    dimensions: Mapping[str, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_update(self) -> Self:
        invalid_names = [
            name for name in self.values
            if type(name) is not str or not name
        ]
        if invalid_names:
            raise ValueError(
                "InputProxy update names must be non-empty exact strings"
            )
        invalid_dimensions = {
            name: size for name, size in self.dimensions.items()
            if (
                type(name) is not str
                or not name
                or type(size) is not int
                or size < 0
            )
        }
        if invalid_dimensions:
            raise ValueError(
                "InputProxy update dimensions require non-empty exact string "
                "names and exact non-negative int sizes"
            )
        object.__setattr__(self, "values", _immutable_dict(self.values))
        object.__setattr__(
            self, "dimensions", _immutable_dict(self.dimensions),
        )
        return self


_INPUT_PROXY_CONTEXT = "hydroforge_input_proxy"


class _InputProxyRemoval(HydroForgeModel):
    """Validated functional removal accepted by :meth:`InputProxy.without`."""

    names: tuple[str, ...]

    @model_validator(mode="after")
    def _validate_removal(self, info: ValidationInfo) -> Self:
        if not self.names:
            raise ValueError(
                "InputProxy.without requires at least one variable"
            )
        if any(type(name) is not str or not name for name in self.names):
            raise ValueError(
                "InputProxy.without names must be non-empty exact strings"
            )
        if len(self.names) != len(set(self.names)):
            raise ValueError("InputProxy.without names must be unique")
        context = info.context
        proxy = (
            context.get(_INPUT_PROXY_CONTEXT)
            if isinstance(context, Mapping) else None
        )
        if proxy is None:
            raise ValueError("InputProxy removal requires proxy context")
        known = proxy.keys().union(proxy.sources)
        missing = set(self.names).difference(known)
        if missing:
            raise ValueError(
                f"InputProxy variable(s) not found: {sorted(missing)}"
            )
        return self


class _InputReadRequest(HydroForgeModel):
    """One orthogonal selection interpreted by resident or lazy storage."""

    name: str
    selector: Any = Field(default_factory=lambda: Ellipsis)
    allow_missing: bool = False

    @model_validator(mode="after")
    def _validate_read(self, info: ValidationInfo):
        if not self.name:
            raise ValueError(
                "input read variable name must be a non-empty string"
            )

        context = info.context
        proxy = (
            context.get(_INPUT_PROXY_CONTEXT)
            if isinstance(context, Mapping) else None
        )
        if proxy is None:
            raise ValueError("input read validation requires proxy context")
        known = self.name in proxy.visible_vars or self.name in proxy.data
        if not known:
            if self.allow_missing:
                return self
            raise ValueError(
                f"InputProxy variable {self.name!r} does not exist"
            )

        raw = self.selector
        selectors = raw if isinstance(raw, tuple) else (raw,)
        if any(value is None for value in selectors):
            raise ValueError(
                "input subset selection does not support new axes"
            )
        for selector in selectors:
            if selector is Ellipsis:
                continue
            if isinstance(selector, slice):
                _normalize_integer_slice(selector)
                continue
            if isinstance(selector, (bool, np.bool_)):
                raise ValueError(
                    "input subset scalar boolean selectors are invalid"
                )
            if isinstance(selector, (int, np.integer)):
                continue
            array = np.asarray(selector)
            if array.ndim == 0 and array.dtype.kind in "iu":
                continue
            if array.ndim == 1 and array.size == 0:
                continue
            if array.ndim != 1 or array.dtype.kind not in "iub":
                raise ValueError(
                    "input subset sequence selectors must be one-dimensional "
                    "integer or boolean arrays"
                )

        shape = proxy._shape_for_trusted(self.name)
        normalized = list(_normalize_netcdf_index(raw, len(shape)))
        for axis, selector in enumerate(normalized):
            if isinstance(selector, slice):
                normalized[axis] = _normalize_integer_slice(selector)
                continue
            index = _as_integer_array(selector, shape[axis])
            if index is not None:
                normalized[axis] = index
                continue
            if _is_scalar_integer(selector):
                integer = int(selector)
                if not -shape[axis] <= integer < shape[axis]:
                    raise ValueError(
                        "input subset integer index exceeds dimension size"
                    )
                normalized[axis] = integer
                continue
            raise ValueError(
                "input subset selectors must be integer scalars, "
                "integer/boolean vectors, or slices"
            )
        object.__setattr__(self, "selector", tuple(normalized))
        return self

    def select_resident(self, value: Any) -> Any:
        """Apply the same per-axis orthogonal contract as NetCDF."""

        return _select_resident_trusted(value, self.selector)


def _select_resident_trusted(value: Any, selector: Any) -> Any:
    """Apply one already normalized orthogonal resident selection."""

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
            sequence_indices.append((
                axis,
                np.arange(
                    *item.indices(value.shape[axis]),
                    dtype=np.int64,
                ),
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


class InputProxy(HydroForgeModel):
    """
    A proxy class for NetCDF input/output.
    Stores data in CPU memory (numpy arrays or torch tensors).
    """

    data: Mapping[str, InputValue]
    attrs: Mapping[str, Any] = Field(default_factory=dict)
    dims: Mapping[str, int] = Field(default_factory=dict)
    lazy: bool = True
    visible_vars: frozenset[str]
    injected_vars: frozenset[str] = Field(default_factory=frozenset)
    sources: Mapping[str, NetCDFInputSource] = Field(default_factory=dict)

    _cache: dict[str, InputValue] = PrivateAttr(default_factory=dict)
    _read_handles: _NetCDFReadHandlePool = PrivateAttr(
        default_factory=_NetCDFReadHandlePool,
    )

    @field_serializer("data")
    def _serialize_data(
        self, value: Mapping[str, InputValue],
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
        invalid_data_names = [
            name for name in self.data
            if type(name) is not str or not name
        ]
        if invalid_data_names:
            raise ValueError(
                "InputProxy data names must be non-empty exact strings"
            )
        if any(type(name) is not str or not name for name in self.attrs):
            raise ValueError(
                "InputProxy attribute names must be non-empty exact strings"
            )
        invalid_dims = {
            name: size for name, size in self.dims.items()
            if (
                type(name) is not str
                or not name
                or type(size) is not int
                or size < 0
            )
        }
        if invalid_dims:
            raise ValueError(
                "InputProxy dimensions must have non-empty exact string "
                "names and exact non-negative int sizes"
            )
        if type(self.lazy) is not bool:
            raise ValueError("InputProxy lazy must be an exact bool")
        invalid_source_names = [
            name for name in self.sources
            if type(name) is not str or not name
        ]
        if invalid_source_names:
            raise ValueError(
                "InputProxy source names must be non-empty exact strings"
            )
        invalid_sources = {
            name: type(source).__name__
            for name, source in self.sources.items()
            if not isinstance(source, NetCDFInputSource)
        }
        if invalid_sources:
            raise ValueError(
                f"InputProxy sources must be NetCDF variable sources: "
                f"{invalid_sources}"
            )
        visible_vars = _name_set(
            self.visible_vars, label="visible_vars",
        )
        object.__setattr__(self, "visible_vars", visible_vars)
        unresolved = visible_vars.difference(self.data).difference(
            self.sources,
        )
        if unresolved:
            raise ValueError(
                "InputProxy visible variables have no resident or lazy source: "
                f"{sorted(unresolved)}"
            )
        lazy_only = visible_vars.difference(self.data)
        if not self.lazy and lazy_only:
            raise ValueError(
                "InputProxy with lazy=False cannot expose source-only "
                f"variables: {sorted(lazy_only)}"
            )
        injected_vars = _name_set(
            self.injected_vars, label="injected_vars",
        )
        invalid_injected = injected_vars.difference(self.data).union(
            injected_vars.difference(visible_vars),
        )
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
        object.__setattr__(self, "dims", _immutable_dict(self.dims))
        object.__setattr__(self, "visible_vars", visible_vars)
        object.__setattr__(self, "injected_vars", injected_vars)
        object.__setattr__(self, "sources", _immutable_dict(self.sources))
        return self

    @property
    def _source_paths(self) -> tuple[Path, ...]:
        """Return distinct physical paths behind lazy variable sources."""

        return tuple(Path(path) for path in dict.fromkeys(
            source.path for source in self.sources.values()
        ))

    def _resident_value(self, name: str) -> InputValue:
        """Return resident storage to an already validated internal path."""

        data = cast(_ResidentInputData, self.data)
        return data._trusted_value(name)

    def _resident_items(self):
        """Iterate resident storage without creating public Tensor copies."""

        data = cast(_ResidentInputData, self.data)
        return data._trusted_items()

    @property
    def file_path(self) -> str | list[str] | None:
        """Compatibility view of :attr:`source_paths`.

        New code should use ``source_paths`` because one proxy can span more
        than one file.  Keeping this read-only view avoids forcing downstream
        parameter-contract errors to lose their source location.
        """

        paths = tuple(str(path) for path in self._source_paths)
        if not paths:
            return None
        return paths[0] if len(paths) == 1 else list(paths)

    def copy(self) -> InputProxy:
        """Return the same validated identity with an independent lazy cache."""

        return self._rebuild()

    def _rebuild(
        self,
        *,
        _owned_data_names: frozenset[str] = frozenset(),
        **updates: Any,
    ) -> Self:
        """Build a derived proxy from validated fields with a fresh cache."""

        payload = {
            name: (
                self.data
                if name == "data"
                else getattr(self, name)
            )
            for name in type(self).model_fields
        }
        payload.update(updates)
        source_data = (
            payload["data"]._trusted_items()
            if isinstance(payload["data"], _ResidentInputData)
            else payload["data"].items()
        )
        payload["data"] = _ResidentInputData({
            name: (
                value
                if name in _owned_data_names
                else _clone_input_value_trusted(value)
            )
            for name, value in source_data
        })
        payload["attrs"] = self.attrs
        payload["dims"] = _immutable_dict(payload["dims"])
        payload["visible_vars"] = frozenset(payload["visible_vars"])
        payload["injected_vars"] = frozenset(payload["injected_vars"])
        payload["sources"] = _immutable_dict(payload["sources"])
        return type(self).model_construct(**payload)

    def updated(
        self,
        *,
        values: Mapping[str, InputValue] | None = None,
        dimensions: Mapping[str, int] | None = None,
    ) -> Self:
        """Return a validated proxy with atomic value/dimension updates."""

        request = _InputProxyUpdate(
            values={} if values is None else values,
            dimensions={} if dimensions is None else dimensions,
        )
        data = dict(self._resident_items())
        data.update(request.values)
        visible = self.visible_vars.union(request.values)
        known = set(self.data).union(self.visible_vars).union(self.sources)
        injected = self.injected_vars.union(
            set(request.values).difference(known),
        )
        dims = dict(self.dims)
        dims.update(request.dimensions)
        return self._rebuild(
            _owned_data_names=frozenset(request.values),
            data=data,
            dims=dims,
            visible_vars=frozenset(visible),
            injected_vars=frozenset(injected),
        )

    def without(self, *names: str) -> Self:
        """Return a validated proxy without the named variables."""

        request = _InputProxyRemoval.model_validate(
            {"names": names},
            context={_INPUT_PROXY_CONTEXT: self},
        )
        removed = set(request.names)
        return self._rebuild(
            data={
                name: value for name, value in self._resident_items()
                if name not in removed
            },
            visible_vars=self.visible_vars.difference(removed),
            injected_vars=self.injected_vars.difference(removed),
            sources={
                name: source for name, source in self.sources.items()
                if name not in removed
            },
        )

    @classmethod
    def from_nc(
        cls,
        file_path: Union[str, Path, List[Union[str, Path]]],
        lazy: bool = False,
        visible_vars: list[str] | set[str] | frozenset[str] | None = None,
        align_on: Optional[str] = None,
        skip_fields: list[str] | set[str] | frozenset[str] | None = None,
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
        declaration = _InputProxyNetCDFDeclaration(
            file_path=file_path,
            lazy=lazy,
            visible_vars=visible_vars,
            align_on=align_on,
            skip_fields=skip_fields,
        )
        plan = declaration.plan
        return cls(
            data=plan.data,
            attrs=plan.attrs,
            dims=plan.dims,
            lazy=declaration.lazy,
            visible_vars=plan.visible_vars,
            sources=plan.sources,
        )

    def _source_for(self, key: str) -> NetCDFInputSource:
        return self.sources[key]

    def _shape_for_trusted(self, key: str) -> tuple[int, ...]:
        if key in self.data:
            value = self._resident_value(key)
            return tuple(value.shape) if hasattr(value, "shape") else ()
        if key in self._cache:
            return tuple(self._cache[key].shape)
        return self.sources[key].shape

    def _read_request(
        self,
        name: str,
        selector: Any = Ellipsis,
        *,
        allow_missing: bool = False,
    ) -> _InputReadRequest:
        return _InputReadRequest.model_validate(
            {
                "name": name,
                "selector": selector,
                "allow_missing": allow_missing,
            },
            context={_INPUT_PROXY_CONTEXT: self},
        )

    def _read_lazy(self, request: _InputReadRequest) -> np.ndarray:
        return _snapshot_input_value(
            self._read_lazy_trusted(request.name, request.selector),
        )

    def _read_lazy_trusted(
        self, name: str, selector: Any,
    ) -> np.ndarray:
        source = self._source_for(name)
        target_path = source.path

        try:
            source.file_identity._verify(target_path)
            with self._read_handles.acquire(target_path) as ds:
                final_indices = source.selectors(selector)
                value = _NetCDFChunkRequest(
                    name=name,
                    dataset=ds,
                    selector=final_indices,
                ).array
            source.file_identity._verify(target_path)
            return value
        except (OSError, RuntimeError) as exc:
            exc.add_note(
                f"while lazily loading {name!r} from {target_path}"
            )
            raise

    @staticmethod
    def _compose_alignment_indices(
        indices: Any, *, ndim: int, axis: int, alignment_idx: np.ndarray,
    ) -> tuple[Any, ...]:
        """Map a reference-order selection onto one source NetCDF variable."""

        del ndim
        selectors = list(indices)
        selectors[axis] = alignment_idx[selectors[axis]]
        return tuple(selectors)

    def get_subset(self, key: str, indices: Any) -> Any:
        """
        Get a subset of a variable.
        If the variable is in memory, slices it.
        If lazy, reads only the requested indices from the file.
        """
        request = self._read_request(key, indices)
        key = request.name
        if key in self.data:
            return _snapshot_input_value(
                request.select_resident(self._resident_value(key)),
            )
        if key in self._cache:
            return _snapshot_input_value(
                request.select_resident(self._cache[key]),
            )

        return self._read_lazy(request)

    def _get_subset_trusted(self, key: str, selector: Any) -> Any:
        """Read a compiler-produced bounded selector without revalidation."""

        shape = self._shape_for_trusted(key)
        normalized = tuple(_normalize_netcdf_index(selector, len(shape)))
        if key in self.data:
            return _select_resident_trusted(
                self._resident_value(key), normalized,
            )
        if key in self._cache:
            return _select_resident_trusted(self._cache[key], normalized)
        return self._read_lazy_trusted(key, normalized)

    def get_var_shape(self, key: str) -> Tuple[int, ...]:
        """
        Get the shape of a variable without loading it fully if possible.
        """
        request = self._read_request(key)
        key = request.name
        # If in memory, return shape
        if key in self.data:
            val = self._resident_value(key)
            # Handle list/scalar or other types if necessary, though data usually is ndarray/tensor
            if hasattr(val, "shape"):
                return tuple(val.shape)
            return ()
        if key in self._cache:
            return tuple(self._cache[key].shape)

        return self._source_for(key).shape

    def _get_var_dtype(self, key: str) -> np.dtype | torch.dtype:
        """Return logical storage dtype without loading a lazy variable."""

        if key in self.data or key in self._cache:
            value = (
                self._resident_value(key)
                if key in self.data else self._cache[key]
            )
            if isinstance(value, torch.Tensor):
                return value.dtype
            return np.asarray(value).dtype

        return self._source_for(key).numpy_dtype


    def _to_nc(
        self,
        file_path: Union[str, Path],
        *,
        netcdf_options: Mapping[str, Any],
    ) -> None:
        """
        Write the stored data to a NetCDF file.
        """
        create_options = netcdf_options
        with _atomic_netcdf_dataset_trusted(file_path) as ds:
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
                    if np.ma.isMaskedArray(data):
                        raise TypeError(
                            f"InputProxy variable {name!r} must not be a "
                            "masked array"
                        )
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
                variable_options = _prepare_netcdf_variable_options_trusted(
                    create_options, dtype=vtype, dimensions=dims, name=name,
                    logical_dtype=logical_dtype,
                )
                var = _create_netcdf_variable_trusted(
                    ds,
                    name,
                    vtype,
                    dims,
                    options=variable_options,
                )
                if logical_dtype is not None:
                    var.setncattr(LOGICAL_DTYPE_ATTR, logical_dtype)
                var[:] = arr_to_write

            # Write variables
            for name in self.keys():
                val = self._get_value_trusted(name)
                _infer_and_write_var(name, val)

    def get(self, key: str, default: Any = None) -> Any:
        request = self._read_request(key, allow_missing=True)
        if key not in self.visible_vars and key not in self.data:
            return default
        return self._get_trusted(request)

    def keys(self) -> Set[str]:
        return set(self.visible_vars).union(self.data)

    def __getitem__(self, key: str) -> Any:
        return self._get_trusted(self._read_request(key))

    def _get_trusted(self, request: _InputReadRequest) -> Any:
        return _snapshot_input_value(
            self._get_value_trusted(request.name),
        )

    def _get_value_trusted(self, key: str) -> Any:
        """Return one schema-owned variable without rebuilding a read query."""

        if key in self.data:
            return self._resident_value(key)
        if key in self._cache:
            return self._cache[key]

        selector = tuple(
            slice(None) for _ in range(len(self.sources[key].shape))
        )
        loaded_data = _preserve_input_value(
            self._read_lazy_trusted(key, selector),
        )
        self._cache[key] = loaded_data
        return loaded_data

    def __contains__(self, key: str) -> bool:
        return key in self.visible_vars

    def close(self) -> None:
        """Close process-local handles used by lazy NetCDF reads."""

        self._read_handles.close()

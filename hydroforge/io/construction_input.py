# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Construction-input NetCDF files: the format of checkpoints.

A construction input holds exactly the values that build a model
(parameters, topology and initial physical state) and nothing else: no time
axis and no run metadata. Its only optional global attribute is the
``hydroforge_options`` fingerprint of the options the values were built with
(``write_construction_input(..., options=...)``); a model materializing from
such a file rejects options that differ from it.  One dimension rule
names every axis.  The leading axis of a variable partitioned by a
coordinate group is named after that group, so the group's coordinate
variable and all its fields share one CF coordinate dimension; every other
axis of ``name`` is ``{name}_dim{axis}``.  Variables are written in the
order of the given mapping.  Writers target the given path directly;
atomic publication belongs to the caller.

Complete variables carry ``hydroforge_complete_data="true"``: their stored
values, including NetCDF fill sentinels, are all data. Framework readers
disable automatic masking for these variables; external readers must honor
the marker or disable masking explicitly. Unmarked input retains CF missing
value semantics.
"""

from __future__ import annotations

import json
import warnings
from collections.abc import Iterable, Mapping, Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any
from unicodedata import normalize

import numpy as np
from netCDF4 import Dataset
from pydantic import Field, InstanceOf, validate_call

from hydroforge.contracts.options import HydroForgeOptionWarning, OptionsConfig
from hydroforge.core.errors import user_stacklevel
from hydroforge.core.naming import validate_netcdf_name
from hydroforge.core.validation import HydroForgeModel
from hydroforge.io.netcdf.encoding import (
    BOOL_NETCDF_READ_DTYPES,
    COMPLETE_DATA_ATTR,
    LOGICAL_DTYPE_ATTR,
    decode_netcdf_logical_array,
    netcdf_dtype_encoding,
    read_netcdf_values,
)
from hydroforge.io.netcdf.options import (
    NetCDFOptions,
    create_netcdf_variable,
    ensure_hdf5_plugins,
    plan_fixed_netcdf_chunks,
    prepare_netcdf_variable_options,
)
from hydroforge.io.netcdf.read import decoded_dtype

_FilePath = Annotated[Path, Field(strict=False)]

OPTIONS_ATTR = "hydroforge_options"
"""Global attribute holding the JSON of ``options.parameter_options()``."""

OPTIONS_INACTIVE_ATTR = "hydroforge_options_inactive"
"""Global attribute listing (JSON) parameter options declared by the writer's
options but not recorded because their ``affects_parameters`` condition (or
relevance) was false when the file was written."""


def _json_value(value: Any) -> Any:
    """Round-trip one option value through JSON (tuples become lists)."""

    return json.loads(json.dumps(value, default=str))


def options_fingerprint(
    options: OptionsConfig, *, opened_modules: Iterable[str] | None = None
) -> str:
    """Return the JSON fingerprint of the options parameter files depend on.

    ``opened_modules`` (the model's selection) decides ``affects_parameters``
    conditions that name modules and drops irrelevant options; see
    ``OptionsConfig.parameter_options``.
    """

    return json.dumps(
        dict(options.parameter_options(opened_modules=opened_modules)),
        sort_keys=True,
        default=str,
    )


def options_attributes(
    options: OptionsConfig, *, opened_modules: Iterable[str] | None = None
) -> dict[str, str]:
    """Return the global attributes recording ``options`` in a file."""

    if opened_modules is not None:
        opened_modules = tuple(opened_modules)
    recorded = options.parameter_options(opened_modules=opened_modules)
    attributes = {
        OPTIONS_ATTR: options_fingerprint(options, opened_modules=opened_modules)
    }
    inactive = sorted(options.parameter_option_paths().difference(recorded))
    if inactive:
        attributes[OPTIONS_INACTIVE_ATTR] = json.dumps(inactive)
    return attributes


def _inactive_paths(attrs: Mapping[str, Any]) -> set[str]:
    recorded = attrs.get(OPTIONS_INACTIVE_ATTR)
    if recorded is None:
        return set()
    try:
        paths = json.loads(str(recorded))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"input global attribute {OPTIONS_INACTIVE_ATTR!r} is not valid JSON"
        ) from error
    if not isinstance(paths, list) or any(type(path) is not str for path in paths):
        raise ValueError(
            f"input global attribute {OPTIONS_INACTIVE_ATTR!r} must be a list of paths"
        )
    return set(paths)


def check_options_fingerprint(
    attrs: Mapping[str, Any],
    options: OptionsConfig,
    *,
    opened_modules: Iterable[str] | None = None,
) -> None:
    """Compare a file's ``hydroforge_options`` with the model's options.

    Options recorded in the file and declared ``affects_parameters`` by the
    model must agree; one ``ValueError`` lists every mismatch. Options present
    on only one side are reported in one ``HydroForgeOptionWarning``. Files
    without the attribute are not checked. Options the model leaves out of
    ``parameter_options(opened_modules=...)`` (``affects_parameters``
    condition false, or irrelevant with ``opened_modules``) are neither
    compared nor reported, nor are options the file lists as inactive.
    """

    recorded = attrs.get(OPTIONS_ATTR)
    if recorded is None:
        return
    try:
        stored = json.loads(str(recorded))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"input global attribute {OPTIONS_ATTR!r} is not valid JSON"
        ) from error
    if not isinstance(stored, dict):
        raise ValueError(f"input global attribute {OPTIONS_ATTR!r} must be an object")
    current = {
        path: _json_value(value)
        for path, value in options.parameter_options(
            opened_modules=opened_modules
        ).items()
    }
    inactive = options.parameter_option_paths().difference(current)
    stored = {path: value for path, value in stored.items() if path not in inactive}
    inactive_in_file = _inactive_paths(attrs)
    mismatches = [
        f"options.{path}: model={current[path]!r}, input file={stored[path]!r}"
        for path in sorted(current.keys() & stored.keys())
        if current[path] != stored[path]
    ]
    if mismatches:
        raise ValueError(
            "the model's options differ from the options the input file was "
            "built with (" + OPTIONS_ATTR + "):\n  - " + "\n  - ".join(mismatches)
        )
    only_file = sorted(stored.keys() - current.keys())
    only_model = sorted(current.keys() - stored.keys() - inactive_in_file)
    if only_file or only_model:
        warnings.warn(
            f"input file {OPTIONS_ATTR} and the model's parameter options name "
            f"different options; recorded only in the file: {only_file}; "
            f"declared only by the model: {only_model}",
            HydroForgeOptionWarning,
            stacklevel=user_stacklevel(),
        )


def construction_dims(
    name: str, ndim: int, *, partition: str | None
) -> tuple[str, ...]:
    """Return the dimension names of one construction-input variable."""

    dims = [f"{name}_dim{axis}" for axis in range(ndim)]
    if partition is not None:
        if not ndim:
            raise ValueError(
                f"construction input {name!r} is partitioned by {partition!r} "
                "but has no axis"
            )
        dims[0] = partition
    return tuple(dims)


@dataclass(frozen=True, slots=True)
class _VariablePlan:
    dtype: np.dtype
    storage: np.dtype
    logical: str | None
    shape: tuple[int, ...]
    dimensions: tuple[str, ...]
    options: Mapping[str, Any]


def _plan_variables(
    layouts: Mapping[str, tuple[np.dtype, tuple[int, ...]]],
    partitions: Mapping[str, str],
    netcdf_options: Mapping[str, Any],
) -> dict[str, _VariablePlan]:
    """Bind the entire file schema before opening a destructive write handle."""

    missing = set(partitions).union(partitions.values()).difference(layouts)
    if missing:
        raise ValueError(
            f"construction partitions refer to missing variables: {sorted(missing)}"
        )
    for group in set(partitions.values()):
        if partitions.get(group) != group:
            raise ValueError(f"construction coordinate {group!r} must partition itself")
        dtype, shape = layouts[group]
        if len(shape) != 1 or dtype.kind not in "iu":
            raise TypeError(
                f"construction coordinate {group!r} must be a one-dimensional integer vector"
            )
    dimensions: dict[str, int] = {}
    canonical_names = {}
    canonical_dimensions = {}
    plans = {}
    for name, (dtype, shape) in layouts.items():
        validate_netcdf_name(name)
        canonical = normalize("NFC", name)
        if canonical in canonical_names:
            raise ValueError(f"NetCDF names normalize to the same spelling: {name!r}")
        canonical_names[canonical] = name
        storage, logical = netcdf_dtype_encoding(dtype)
        if not (
            storage.kind in "iu"
            and storage.itemsize in {1, 2, 4, 8}
            or storage.kind == "f"
            and storage.itemsize in {4, 8}
            or storage == np.dtype("S1")
            or storage.kind == "U"
        ):
            raise TypeError(
                f"construction input {name!r} has unsupported NetCDF dtype {dtype}"
            )
        dims = construction_dims(name, len(shape), partition=partitions.get(name))
        for dim, size in zip(dims, shape, strict=True):
            validate_netcdf_name(dim)
            canonical = normalize("NFC", dim)
            previous_name = canonical_dimensions.setdefault(canonical, dim)
            if previous_name != dim:
                raise ValueError(
                    f"NetCDF dimensions normalize to the same spelling: {dim!r}"
                )
            previous = dimensions.setdefault(dim, size)
            if previous != size:
                raise ValueError(
                    f"construction input {name!r} has length {size} along {dim!r}, which already has length {previous}"
                )
        options = prepare_netcdf_variable_options(
            plan_fixed_netcdf_chunks(netcdf_options, dtype=storage, shape=shape),
            dtype=storage,
            dimensions=dims,
            name=name,
            logical_dtype=logical,
            shape=shape,
        )
        plans[name] = _VariablePlan(dtype, storage, logical, shape, dims, options)
    return plans


def _create(dataset: Dataset, name: str, plan: _VariablePlan):
    for dim, size in zip(plan.dimensions, plan.shape, strict=True):
        if dim not in dataset.dimensions:
            dataset.createDimension(dim, size)
    variable = create_netcdf_variable(
        dataset, name, plan.storage, plan.dimensions, options=plan.options
    )
    variable.setncattr(COMPLETE_DATA_ATTR, "true")
    if plan.logical is not None:
        variable.setncattr(LOGICAL_DTYPE_ATTR, plan.logical)
    return variable


def _write(
    dataset: Dataset, name: str, values: np.ndarray, plan: _VariablePlan
) -> None:
    variable = _create(dataset, name, plan)
    stored = values.astype(plan.storage, copy=False)
    if values.ndim:
        variable[:] = stored
    else:
        variable.assignValue(stored)


@validate_call(config=HydroForgeModel.model_config)
def write_construction_input(
    path: _FilePath,
    values: Mapping[str, Any],
    *,
    partitions: Mapping[str, str],
    netcdf_options: NetCDFOptions,
    options: InstanceOf[OptionsConfig] | None = None,
    opened_modules: tuple[str, ...] | None = None,
) -> None:
    """Write host arrays and scalars as one construction-input file.

    ``partitions`` maps each partitioned variable, including the group's
    coordinate variable, to its coordinate group; ``netcdf_options`` are
    ``Dataset.createVariable`` options. ``options`` (the model's options the
    values were built with) records the ``hydroforge_options`` fingerprint of
    its parameter options as a global attribute; pass the model's
    ``opened_modules`` to decide ``affects_parameters`` conditions that name
    modules (see :func:`options_attributes`).
    """

    arrays = {}
    for name, value in values.items():
        if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
            raise ValueError(f"construction input {name!r} contains missing values")
        arrays[name] = np.asarray(value)
    plans = _plan_variables(
        {name: (value.dtype, value.shape) for name, value in arrays.items()},
        partitions,
        netcdf_options,
    )
    ensure_hdf5_plugins()
    if options is None and opened_modules is not None:
        raise ValueError("opened_modules requires options")
    attributes = (
        {}
        if options is None
        else options_attributes(options, opened_modules=opened_modules)
    )
    with Dataset(path, "w", format="NETCDF4") as dataset:
        for name, value in attributes.items():
            dataset.setncattr(name, value)
        for name, value in arrays.items():
            _write(dataset, name, value, plans[name])


def _read(part: Path, variable: Any, name: str) -> np.ndarray:
    raw = read_netcdf_values(variable)
    if np.ma.isMaskedArray(raw) and np.any(np.ma.getmaskarray(raw)):
        raise ValueError(
            f"construction part {str(part)!r} variable {name!r} contains missing values"
        )
    values = np.asarray(decode_netcdf_logical_array(variable, raw, name=name))
    if variable.dtype is str:
        # Variable-length strings read as objects; restore the written kind.
        values = values.astype(str)
    return values


def _same_layout(values: np.ndarray, plan: _VariablePlan) -> bool:
    # A string's width is a property of its values, not of the variable.
    same_dtype = (
        values.dtype == plan.dtype or values.dtype.kind == plan.dtype.kind == "U"
    )
    return same_dtype and values.shape[1:] == plan.shape[1:]


@validate_call(config=HydroForgeModel.model_config)
def merge_construction_parts(
    path: _FilePath,
    parts: Annotated[Sequence[_FilePath], Field(min_length=1)],
    *,
    partitions: Mapping[str, str],
    netcdf_options: NetCDFOptions,
    options: InstanceOf[OptionsConfig] | None = None,
    opened_modules: tuple[str, ...] | None = None,
) -> None:
    """Concatenate rank-local construction parts into one file.

    Every partitioned variable occurs in every part and is concatenated along
    its group axis in part order; the other variables come from part zero
    only, whose variable order the merged file keeps.  Coordinate IDs must be
    unique across the parts. ``options`` and ``opened_modules`` record the
    merged file's option fingerprint as in :func:`write_construction_input`.
    """

    input_identities = set()
    for part in parts:
        status = part.stat()
        identity = (status.st_dev, status.st_ino)
        if identity in input_identities:
            raise ValueError("construction input parts must identify distinct files")
        input_identities.add(identity)
        if path.resolve() == part.resolve() or (path.exists() and path.samefile(part)):
            raise ValueError("construction output must not overwrite an input part")
    groups = sorted(set(partitions.values()))
    for group in groups:
        if partitions.get(group) != group:
            raise ValueError(f"construction coordinate {group!r} must partition itself")
    ensure_hdf5_plugins()
    with ExitStack() as stack:
        datasets = [stack.enter_context(Dataset(part, "r")) for part in parts]
        coordinates: dict[str, list[np.ndarray]] = {group: [] for group in groups}
        layouts = {}
        signatures = {}
        for index, (part, dataset) in enumerate(zip(parts, datasets, strict=True)):
            if dataset.ncattrs():
                raise ValueError(
                    f"construction part {str(part)!r} must not contain global "
                    f"attributes: {sorted(dataset.ncattrs())}"
                )
            missing = set(partitions).difference(dataset.variables)
            if missing:
                raise ValueError(
                    f"construction part {str(part)!r} is missing partitioned "
                    f"variables: {sorted(missing)}"
                )
            unexpected = set(dataset.variables).difference(partitions)
            if index and unexpected:
                raise ValueError(
                    f"construction part {str(part)!r} of rank {index} contains "
                    f"unpartitioned variables: {sorted(unexpected)}"
                )
            for name, variable in dataset.variables.items():
                dtype = decoded_dtype(variable)
                logical = getattr(variable, LOGICAL_DTYPE_ATTR, None)
                if logical == "bool":
                    if np.dtype(variable.dtype) not in BOOL_NETCDF_READ_DTYPES:
                        raise TypeError(
                            f"boolean NetCDF variable {name!r} must use i1/u1 storage"
                        )
                    dtype = np.dtype(bool)
                signature = (dtype, variable.shape[1:], logical)
                if index == 0:
                    signatures[name] = signature
                    layouts[name] = (dtype, variable.shape)
                elif signature != signatures[name]:
                    raise TypeError(
                        f"construction part {str(part)!r} variable {name!r} has "
                        f"dtype, trailing shape and logical dtype {signature}, expected {signatures[name]}"
                    )
            for group in groups:
                ids = _read(part, dataset.variables[group], group)
                if ids.ndim != 1 or ids.dtype.kind not in "iu":
                    raise TypeError(
                        f"construction coordinate {group!r} must be a "
                        "one-dimensional integer vector"
                    )
                coordinates[group].append(ids)
            for name, group in partitions.items():
                shape = dataset.variables[name].shape
                length = coordinates[group][index].size
                if not shape or shape[0] != length:
                    raise ValueError(
                        f"construction part {str(part)!r} variable {name!r} does "
                        f"not match the length {length} of coordinate group {group!r}"
                    )
        for group, pieces in coordinates.items():
            ids = np.concatenate(pieces)
            if np.unique(ids).size != ids.size:
                raise ValueError(
                    f"construction coordinate {group!r} contains duplicate IDs "
                    "across parts"
                )

        for name, group in partitions.items():
            dtype, shape = layouts[name]
            layouts[name] = (
                dtype,
                (sum(piece.size for piece in coordinates[group]), *shape[1:]),
            )
        plans = _plan_variables(layouts, partitions, netcdf_options)
        if options is None and opened_modules is not None:
            raise ValueError("opened_modules requires options")
        attributes = (
            {}
            if options is None
            else options_attributes(options, opened_modules=opened_modules)
        )
        with Dataset(path, "w", format="NETCDF4") as merged:
            for name, value in attributes.items():
                merged.setncattr(name, value)
            for name, plan in plans.items():
                group = partitions.get(name)
                if group is None:
                    _write(
                        merged,
                        name,
                        _read(parts[0], datasets[0].variables[name], name),
                        plan,
                    )
                    continue
                variable = _create(merged, name, plan)
                offset = 0
                for index, (part, dataset) in enumerate(
                    zip(parts, datasets, strict=True)
                ):
                    values = (
                        coordinates[name][index]
                        if name == group
                        else _read(part, dataset.variables[name], name)
                    )
                    # Actual decode is a new value boundary even after schema preflight.
                    if not _same_layout(values, plan):
                        raise TypeError(
                            f"construction part {str(part)!r} variable {name!r} changed decoded layout"
                        )
                    end = offset + values.shape[0]
                    variable[offset:end, ...] = values.astype(plan.storage, copy=False)
                    offset = end

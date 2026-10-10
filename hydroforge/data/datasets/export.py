# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""NetCDF exports of forcing datasets and the CaMa mapping-table generator.

Time-series exports are single-rank files of the rank-output schema
(:mod:`hydroforge.io.rank_output.schema`), readable by
:class:`~hydroforge.data.datasets.ExportedDataset` and
:class:`~hydroforge.io.MultiRankStatsReader`.  Every export reads at the source
cadence and excludes spin-up chunks.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from contextlib import ExitStack
from functools import partial
from itertools import chain
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

import netCDF4 as nc
import numpy as np
import torch
from pydantic import AfterValidator, BeforeValidator, Field, InstanceOf, validate_call
from tqdm import tqdm

from hydroforge.core.arrays import canonical_float64, canonical_floating_array
from hydroforge.core.errors import cleanup_on_exit
from hydroforge.core.naming import sanitize_symbol, validate_netcdf_name
from hydroforge.core.validation import HydroForgeModel
from hydroforge.data.datasets.base import ForcingDataset, TorchDevice
from hydroforge.data.datasets.exported import ExportedDataset
from hydroforge.data.datasets.space import GridSpace
from hydroforge.io.netcdf.encoding import COMPLETE_DATA_ATTR
from hydroforge.io.netcdf.options import (
    NetCDFOptions,
    create_netcdf_variable,
    default_netcdf_options,
    prepare_netcdf_variable_options,
)
from hydroforge.io.netcdf.write import atomic_netcdf_dataset
from hydroforge.io.rank_output.schema import (
    COMMITTED_STEPS_ATTR,
    POINT_DIM,
    TIME_DIM,
    RankFileHeader,
    create_time_axis,
    rank_file_name,
    write_point_coordinate,
)
from hydroforge.mapping.aggregation import build_cama_mapping
from hydroforge.mapping.table import normalize_target_weights
from hydroforge.parallel.distributed import is_rank_zero

logger = logging.getLogger(__name__)

_EXPORT = validate_call(config=HydroForgeModel.model_config)
_COORDINATE = "catchment_id"


def _output_name(value: str, *, label: str) -> str:
    if not value or sanitize_symbol(value) != value:
        raise ValueError(f"{label} must be one safe NetCDF/file component")
    if value in {_COORDINATE, TIME_DIM, "quantile"}:
        raise ValueError(f"{label} conflicts with a reserved export coordinate")
    return value


_SafeName = Annotated[str, AfterValidator(partial(_output_name, label="name"))]


def _quantile_levels(value: np.ndarray) -> np.ndarray:
    if np.ma.isMaskedArray(value):
        raise ValueError("quantiles must not be a masked array")
    array = np.asarray(value)
    if array.ndim != 1 or array.size == 0:
        raise ValueError("quantiles must be a non-empty one-dimensional array")
    if array.dtype.kind not in {"f", "i", "u"}:
        raise ValueError("quantiles must contain real numeric values")
    if not np.isfinite(array).all() or np.any((array < 0) | (array > 1)):
        raise ValueError("quantiles must lie within [0, 1]")
    result = canonical_float64(array, label="quantiles")
    if np.any(np.diff(result) <= 0):
        raise ValueError("quantiles must be strictly increasing")
    return result


def _source_nan_mask(value: np.ndarray) -> np.ndarray:
    if np.ma.isMaskedArray(value) or value.dtype != np.dtype(np.bool_):
        raise ValueError("source_nan_mask must be an unmasked boolean array")
    return value


def _metadata(
    value: str | Mapping[str, str] | None,
    *,
    label: str,
    names: tuple[str, ...],
    default: Callable[[str], str],
) -> Mapping[str, str]:
    """One value per output name from a string, a complete mapping or ``None``."""

    if value is None:
        return {name: default(name) for name in names}
    if isinstance(value, str):
        return dict.fromkeys(names, value)
    if set(value) != set(names):
        raise ValueError(f"{label} mapping keys must be exactly {list(names)}")
    return {name: value[name] for name in names}


def _float64_weights(mapping: torch.Tensor, device: torch.device) -> torch.Tensor:
    """The ``(targets, sources)`` mapping as coalesced float64 COO on ``device``."""

    return mapping.to_sparse_coo().to(device=device, dtype=torch.float64).coalesce()


def _main_blocks(dataset: ForcingDataset):
    """Source-cadence values of each main-period chunk."""

    plan = dataset.chunk_plan
    for index in range(plan.num_spinup_chunks, len(plan)):
        yield plan.chunks[index], dataset._read_source(index)


def _named_blocks(read: Any, fallback: str) -> dict[str, np.ndarray]:
    """Flatten actual output keys; a single array keeps the caller's name."""

    blocks = {}

    pending = [((), read)]
    while pending:
        path, value = pending.pop()
        if isinstance(value, Mapping):
            pending.extend(
                ((*path, key), child) for key, child in reversed(tuple(value.items()))
            )
        else:
            name = "_".join(path) if path else fallback
            if name in blocks:
                raise ValueError(f"export output paths collide at {name!r}")
            blocks[name] = value

    return blocks


def _declared_names(dataset: ForcingDataset, fallback: str) -> tuple[str, ...] | None:
    """Flatten builtin output declarations without reading source payloads."""
    from hydroforge.data.datasets.base import DatasetExpression
    from hydroforge.data.datasets.composite import MultiVariableDataset
    from hydroforge.data.datasets.netcdf import NetCDFDataset

    def paths(source):
        if isinstance(source, MultiVariableDataset):
            result = []
            for name, child in source.datasets.items():
                nested = paths(child)
                if nested is None:
                    return None
                result.extend((name, *path) for path in nested)
            return tuple(result)
        if isinstance(source, DatasetExpression):
            schemas = [paths(child) for _label, child in source._children()]
            if any(schema is None for schema in schemas):
                return None
            named = [schema for schema in schemas if schema != ((),)]
            if named and any(set(schema) != set(named[0]) for schema in named[1:]):
                raise ValueError(
                    "dataset expression operands have different output names"
                )
            return named[0] if named else ((),)
        if isinstance(source, (NetCDFDataset, ExportedDataset)):
            aggregation = source.time_aggregation
            return (
                tuple((name,) for name in aggregation)
                if isinstance(aggregation, Mapping)
                else ((),)
            )
        return None

    schema = paths(dataset)
    if schema is None:
        return None
    names = tuple("_".join(path) if path else fallback for path in schema)
    _validate_names(names)
    return names


def _validate_names(names: tuple[str, ...]) -> None:
    if len(set(names)) != len(names):
        raise ValueError("export output paths collide")
    for name in names:
        validate_netcdf_name(name)
        _output_name(name, label="export variable name")


@_EXPORT
def export_climatology(
    dataset: InstanceOf[ForcingDataset],
    mapping: InstanceOf[torch.Tensor],
    out_path: Annotated[Path, Field(strict=False)],
    *,
    var_name: _SafeName,
    dtype: Literal["float32", "float64"] = "float32",
    netcdf_options: NetCDFOptions = Field(default_factory=default_netcdf_options),
    device: TorchDevice = torch.device("cpu"),
    units: str | Mapping[str, str] = "m3/s",
    description: str | Mapping[str, str] | None = None,
) -> Path:
    """Write the main-period time mean of a mapped view per target.

    Like the Fortran routing models, every source step is mapped and summed,
    then divided by the step count.  The file has one ``saved_points``
    dimension with a ``catchment_id`` coordinate.
    """

    catchment_ids = dataset._require_mapping(mapping, caller="export")
    declared = _declared_names(dataset, var_name)
    if declared is not None:
        _metadata(units, label="units", names=declared, default=lambda _: "")
        _metadata(description, label="description", names=declared, default=str)
    prepare_netcdf_variable_options(
        netcdf_options,
        dtype="f4" if dtype == "float32" else "f8",
        dimensions=(POINT_DIM,),
        name=var_name,
        shape=(catchment_ids.size,),
    )
    progress = tqdm(
        _main_blocks(dataset),
        total=len(dataset.chunk_plan) - dataset.chunk_plan.num_spinup_chunks,
        desc="Computing climatology",
        unit="chunk",
    )
    with cleanup_on_exit("climatology progress", (progress.close,)):
        chunks = iter(progress)
        first = next(chunks, None)
        if first is None:
            raise RuntimeError("No valid timesteps found — cannot compute climatology.")
        names = tuple(_named_blocks(first[1], var_name))
        _validate_names(names)
        descriptions = _metadata(
            description,
            label="description",
            names=names,
            default=lambda name: f"Time-averaged {name}",
        )
        units_by_name = _metadata(
            units, label="units", names=names, default=lambda _: ""
        )
        dtype_nc = "f4" if dtype == "float32" else "f8"
        create_options = {
            name: prepare_netcdf_variable_options(
                netcdf_options, dtype=dtype_nc, dimensions=(POINT_DIM,), name=name
            )
            for name in names
        }
        weights = _float64_weights(mapping, device)
        total_steps = 0
        accumulators = {
            name: torch.zeros(catchment_ids.size, dtype=torch.float64, device=device)
            for name in names
        }
        stream = chain(iter((first,)), chunks)
        first = None
        for chunk, read in stream:
            blocks = _named_blocks(read, var_name)
            if set(blocks) != set(names):
                raise ValueError("export source output names changed between chunks")
            for name, block in blocks.items():
                values = torch.as_tensor(
                    np.ascontiguousarray(block, dtype=np.float64),
                    dtype=torch.float64,
                    device=device,
                )
                accumulators[name] += torch.sparse.mm(weights, values.T).sum(dim=1)
            total_steps += chunk.length
    if total_steps == 0:
        raise RuntimeError("No valid timesteps found — cannot compute climatology.")
    means = {
        name: canonical_floating_array(
            (value / total_steps).cpu().numpy(),
            dtype=dtype,
            label=f"climatology {name!r}",
        )
        for name, value in accumulators.items()
    }
    if is_rank_zero():
        logger.info("Climatology averaged over %d timesteps", total_steps)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_netcdf_dataset(out_path, format="NETCDF4") as output:
        output.setncattr("title", f"Climatology ({var_name})")
        output.setncattr("total_timesteps", total_steps)
        output.createDimension(POINT_DIM, catchment_ids.size)
        write_point_coordinate(output, _COORDINATE, catchment_ids)
        for name in names:
            variable = create_netcdf_variable(
                output, name, dtype_nc, (POINT_DIM,), options=create_options[name]
            )
            variable.setncattr(COMPLETE_DATA_ATTR, "true")
            variable[:] = means[name]
            variable.setncattr(
                "description",
                f"{descriptions[name]} over {total_steps} steps"
                if description is None
                else descriptions[name],
            )
            variable.setncattr("units", units_by_name[name])
    if is_rank_zero():
        logger.info("Saved climatology to %s", out_path)
    return out_path


@_EXPORT
def export_catchment_data(
    dataset: InstanceOf[ForcingDataset],
    mapping: InstanceOf[torch.Tensor],
    out_dir: Annotated[Path, Field(strict=False)],
    *,
    var_name: _SafeName = "var",
    filename: _SafeName | Mapping[str, _SafeName] | None = None,
    dtype: Literal["float32", "float64"] = "float32",
    netcdf_options: NetCDFOptions = Field(default_factory=default_netcdf_options),
    normalized: bool = False,
    device: TorchDevice = torch.device("cpu"),
    split_by_year: bool = False,
    units: str | Mapping[str, str] = "m3/s",
    description: str | Mapping[str, str] | None = None,
) -> Path | list[Path] | dict[str, Path] | dict[str, list[Path]]:
    """Map a view onto its targets and write ``{filename}_rank0[_{year}].nc``.

    The files hold an unlimited ``time`` axis in days, the ``catchment_id``
    coordinate on ``saved_points`` and one ``(time, saved_points)`` variable
    per output; a ``time_aggregation`` mapping writes one file set per
    aggregation.  ``normalized`` scales each target's weights to sum to one.
    Use a CUDA ``device`` for the sparse products of large mappings.
    """

    if netcdf_options.get("contiguous") is True:
        raise ValueError(
            "streaming catchment output has unlimited time and cannot use contiguous=True"
        )
    catchment_ids = dataset._require_mapping(mapping, caller="export")
    declared = _declared_names(dataset, var_name)
    if declared is not None:
        _metadata(units, label="units", names=declared, default=lambda _: "")
        _metadata(description, label="description", names=declared, default=str)
        filenames = _metadata(filename, label="filename", names=declared, default=str)
        if len(set(filenames.values())) != len(filenames):
            raise ValueError("output filenames must be unique across variables")
    prepare_netcdf_variable_options(
        netcdf_options,
        dtype="f4" if dtype == "float32" else "f8",
        dimensions=(TIME_DIM, POINT_DIM),
        name=var_name,
        shape=(None, catchment_ids.size),
    )
    writers: dict[str, tuple[Any, Any, Any]] = {}
    writer_stack: ExitStack | None = None
    write_index = 0

    def close_writers(error: BaseException | None = None) -> None:
        nonlocal writers, writer_stack
        closing, stack = writers, writer_stack
        writers, writer_stack = {}, None
        if stack is None:
            return
        if error is not None:
            stack.__exit__(type(error), error, error.__traceback__)
            return
        try:
            for output, time_variable, _variable in closing.values():
                output.setncattr(COMMITTED_STEPS_ATTR, len(time_variable))
                output.sync()
        except BaseException as commit_error:
            with cleanup_on_exit(
                "export commit",
                (
                    partial(
                        stack.__exit__,
                        type(commit_error),
                        commit_error,
                        commit_error.__traceback__,
                    ),
                ),
            ):
                raise
        else:
            stack.close()

    progress = tqdm(total=dataset.num_main_source_steps, desc="Exporting", unit="step")
    failure = None

    def close_progress() -> None:
        nonlocal failure
        try:
            progress.close()
        except BaseException as error:
            if failure is None:
                failure = error
            raise

    with (
        cleanup_on_exit("export writers", (lambda: close_writers(failure),)),
        cleanup_on_exit("export progress", (close_progress,)),
    ):
        try:
            chunks = iter(_main_blocks(dataset))
            first = next(chunks)
            named = isinstance(first[1], Mapping)
            names = tuple(_named_blocks(first[1], var_name))
            _validate_names(names)
            aggregation = getattr(dataset, "time_aggregation", None)
            if isinstance(aggregation, Mapping):
                methods = dict(aggregation)
            else:
                methods = dict.fromkeys(names, aggregation)
            for name in names:
                _output_name(name, label="time_aggregation output name")
            filenames = _metadata(filename, label="filename", names=names, default=str)
            if len(set(filenames.values())) != len(filenames):
                raise ValueError("output filenames must be unique across variables")
            descriptions = _metadata(
                description,
                label="description",
                names=names,
                default=lambda name: (
                    f"Catchment-aggregated {name} ({methods[name]})"
                    if methods[name] is not None
                    else f"Catchment-aggregated {name}"
                ),
            )
            units_by_name = _metadata(
                units, label="units", names=names, default=lambda _: ""
            )
            dtype_nc = "f4" if dtype == "float32" else "f8"
            create_options = {
                name: prepare_netcdf_variable_options(
                    netcdf_options,
                    dtype=dtype_nc,
                    dimensions=(TIME_DIM, POINT_DIM),
                    name=name,
                )
                for name in names
            }

            weights = _float64_weights(mapping, device)
            if normalized:
                # Each target (mapping row) sums to one.
                indices = weights.indices()
                normalized_values = normalize_target_weights(
                    indices[0].cpu().numpy(),
                    weights.values().cpu().numpy(),
                    catchment_ids.size,
                )
                weights = torch.sparse_coo_tensor(
                    indices,
                    torch.from_numpy(normalized_values).to(device),
                    weights.size(),
                    dtype=torch.float64,
                    device=device,
                ).coalesce()

            out_dir.mkdir(parents=True, exist_ok=True)
            calendar = dataset.simulation_schedule.calendar
            header = RankFileHeader(rank=0, world_size=1, run_id=str(uuid4()))

            def create(stack: ExitStack, path: Path, name: str) -> tuple[Any, Any, Any]:
                output = stack.enter_context(
                    atomic_netcdf_dataset(path, format="NETCDF4")
                )
                output.setncattr("title", f"Aggregated catchment data ({name})")
                header.write(output)
                if methods[name] is not None:
                    output.setncattr("time_aggregation", methods[name])
                output.createDimension(TIME_DIM, None)
                output.createDimension(POINT_DIM, catchment_ids.size)
                time_variable = create_time_axis(output, calendar=calendar)
                write_point_coordinate(output, _COORDINATE, catchment_ids)
                variable = create_netcdf_variable(
                    output,
                    name,
                    dtype_nc,
                    (TIME_DIM, POINT_DIM),
                    options=create_options[name],
                )
                variable.setncattr(COMPLETE_DATA_ATTR, "true")
                variable.setncattr("description", descriptions[name])
                variable.setncattr("units", units_by_name[name])
                return output, time_variable, variable

            def open_writers(year: int | None = None) -> None:
                nonlocal write_index, writer_stack
                close_writers()
                writer_stack = ExitStack()
                try:
                    for name in names:
                        path = out_dir / rank_file_name(filenames[name], 0, year)
                        writers[name] = create(writer_stack, path, name)
                        created[name].append(path)
                except BaseException as error:
                    with cleanup_on_exit(
                        "export setup", (partial(close_writers, error),)
                    ):
                        raise
                write_index = 0

            created: dict[str, list[Path]] = {name: [] for name in names}
            if not split_by_year:
                open_writers()
            current_year = None
            stream = chain(iter((first,)), chunks)
            first = None
            for chunk, read in stream:
                blocks = _named_blocks(read, var_name)
                if set(blocks) != set(names):
                    raise ValueError(
                        "export source output names changed between chunks"
                    )
                mapped = {}
                for name, block in blocks.items():
                    values = torch.as_tensor(
                        np.ascontiguousarray(block, dtype=np.float64),
                        dtype=torch.float64,
                        device=device,
                    )
                    mapped[name] = canonical_floating_array(
                        torch.sparse.mm(weights, values.T).T.contiguous().cpu().numpy(),
                        dtype=dtype,
                        label=f"aggregated variable {name!r} at chunk {chunk.index}",
                    )
                # Write maximal same-file runs as blocks.
                times = chunk.source_times()
                start = 0
                while start < chunk.length:
                    stop = chunk.length
                    if split_by_year:
                        if times[start].year != current_year:
                            current_year = times[start].year
                            open_writers(current_year)
                        stop = start + 1
                        while stop < chunk.length and times[stop].year == current_year:
                            stop += 1
                    _output, first_time, _variable = next(iter(writers.values()))
                    time_values = nc.date2num(
                        times[start:stop],
                        units=first_time.getncattr("units"),
                        calendar=first_time.getncattr("calendar"),
                    )
                    end = write_index + stop - start
                    for name in names:
                        _output, time_variable, variable = writers[name]
                        variable[write_index:end, :] = mapped[name][start:stop, :]
                        time_variable[write_index:end] = time_values
                    progress.update(stop - start)
                    write_index = end
                    start = stop
        except BaseException as error:
            failure = error
            raise

    if named:
        if split_by_year:
            return created
        return {name: paths[0] for name, paths in created.items()}
    paths = created[names[0]]
    return paths if split_by_year else paths[0]


@_EXPORT
def export_quantiles(
    dataset: InstanceOf[ExportedDataset],
    out_path: Annotated[Path, Field(strict=False)],
    *,
    quantiles: Annotated[np.ndarray, BeforeValidator(_quantile_levels)] = (
        0.0,
        0.1,
        0.25,
        0.5,
        0.75,
        0.9,
        1.0,
    ),
    var_name: Annotated[str, Field(min_length=1)] | None = None,
    dtype: Literal["float32", "float64"] = "float32",
    netcdf_options: NetCDFOptions = Field(default_factory=default_netcdf_options),
    max_buffer_mb: Annotated[float, Field(gt=0, allow_inf_nan=False)] = 4096.0,
) -> Path:
    """Write per-point temporal quantiles of the main period.

    The file has dimensions ``quantile`` and ``saved_points`` with the
    ``catchment_id`` coordinate in the dataset's (selected) order; that
    coordinate is an identity, so the selection must not repeat IDs.  Exact
    quantiles need each point's full series: when the estimated source
    working set (three arrays on the expanded source time axis, at the
    widest source or output width) exceeds ``max_buffer_mb``, points are
    processed in column batches.  A budget below one column is rejected
    before reading.  Resident values are reused without new reads.
    Quantiles describe the unshifted source-time population, not model-step
    resampling. Shifted views must be selected again without time shifts.
    """

    if dataset._shift is not None:
        raise ValueError("export_quantiles requires an unshifted source-time view")
    name = dataset.var_name if var_name is None else var_name
    dtype_nc = "f4" if dtype == "float32" else "f8"
    space = dataset.space
    point_ids = space.selected_ids
    columns = point_ids.size
    # The written coordinate is an identity; a query may repeat IDs.
    if np.unique(point_ids).size != columns:
        raise ValueError("export_quantiles requires unique selected IDs")
    steps = dataset.num_main_source_steps
    _declared_names(dataset, name)
    prepare_netcdf_variable_options(
        netcdf_options,
        dtype=dtype_nc,
        dimensions=("quantile", POINT_DIM),
        name=name,
        shape=(quantiles.size, columns),
    )
    budget = max_buffer_mb * 1024 * 1024
    main_ops = None
    resident = dataset._resident
    if resident is not None:
        full_size = dataset._cache_nbytes(resident.main)
        fits = True
    else:
        # The peak read is on the expanded source axis, before aggregation.
        # Reading, concatenation and exact selection can briefly hold three
        # arrays of the widest element.
        main_ops = dataset._store.timeline.operations(dataset._plan.domain.times())
        source_rows = sum(len(rows) for _key, rows in main_ops)
        element_bytes = 3 * dataset._source_element_bytes(main_ops)
        outputs = (
            len(dataset.time_aggregation)
            if isinstance(dataset.time_aggregation, Mapping)
            else 1
        )
        column_bytes = source_rows * element_bytes
        if outputs > 1:
            column_bytes += steps * outputs * (8 + np.dtype(dataset.out_dtype).itemsize)
        full_size = columns * column_bytes
        fits = full_size <= budget
    if not fits:
        if budget < column_bytes:
            raise ValueError(
                "max_buffer_mb is too small for one quantile source column; "
                f"requires at least {column_bytes} bytes "
                f"({column_bytes / 1024**2:g} MiB)"
            )
        batch = int(budget / column_bytes)
        if is_rank_zero():
            logger.info(
                "Exported dataset %.1f GB exceeds %.0f MiB buffer; "
                "processing %d catchments in %d batches of %d",
                full_size / 1e9,
                max_buffer_mb,
                columns,
                (columns + batch - 1) // batch,
                batch,
            )

    if fits:
        dataset.load_to_memory()
        first = dataset._resident.main
    else:
        positions = (
            np.arange(columns, dtype=np.int64)
            if space.selection is None
            else space.selection
        )
        first = dataset._read_values(main_ops, positions[:batch], None)
    blocks = _named_blocks(first, name)
    names = tuple(blocks)
    _validate_names(names)
    create_options = {
        key: prepare_netcdf_variable_options(
            netcdf_options, dtype=dtype_nc, dimensions=("quantile", POINT_DIM), name=key
        )
        for key in names
    }

    def write_quantiles(variables, values_by_name, start, stop, *, owned):
        if set(values_by_name) != set(names):
            raise ValueError("export source output names changed between batches")
        for key, values in values_by_name.items():
            values = values[:steps]
            if not np.isfinite(values).all():
                raise ValueError("quantile source contains non-finite values")
            variables[key][:, start:stop] = canonical_floating_array(
                np.quantile(values, quantiles, axis=0, overwrite_input=owned),
                dtype=dtype,
                label=f"quantile result {key!r}",
            )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_netcdf_dataset(out_path, format="NETCDF4") as output:
        output.createDimension("quantile", quantiles.size)
        output.createDimension(POINT_DIM, columns)
        levels = output.createVariable("quantile", "f8", ("quantile",))
        levels[:] = quantiles
        levels.long_name = "quantile level"
        write_point_coordinate(output, _COORDINATE, point_ids)
        variables = {}
        for key in names:
            variable = create_netcdf_variable(
                output,
                key,
                dtype_nc,
                ("quantile", POINT_DIM),
                options=create_options[key],
            )
            variable.setncattr(COMPLETE_DATA_ATTR, "true")
            variable.long_name = f"{key} quantile values"
            variables[key] = variable
        if fits:
            write_quantiles(variables, blocks, 0, columns, owned=False)
        else:
            # Exact quantiles need every time step of each batch of points.
            for start in range(0, columns, batch):
                stop = min(start + batch, columns)
                if start:
                    blocks = _named_blocks(
                        dataset._read_values(main_ops, positions[start:stop], None),
                        name,
                    )
                write_quantiles(variables, blocks, start, stop, owned=True)
                # Release this batch before the next read so the working-set
                # estimate stays conservative across batch boundaries.
                blocks.clear()
                first = None
    if is_rank_zero():
        logger.info(
            "Saved quantiles to %s: levels=%s, shape=(%d, %d)",
            out_path,
            quantiles.tolist(),
            quantiles.size,
            columns,
        )
    return out_path


@_EXPORT
def generate_mapping_table(
    dataset: InstanceOf[ForcingDataset],
    map_dir: Annotated[Path, Field(strict=False)],
    out_path: Annotated[Path, Field(strict=False)],
    *,
    mapinfo_txt: str = "location.txt",
    hires_tag: str | None = "1min",
    lowres_idx_precision: str = "<i4",
    hires_idx_precision: str = "<i2",
    map_precision: str = "<f4",
    parameter_nc: Annotated[Path, Field(strict=False)] | None = None,
    allow_oob_zero: bool = False,
    source_nan_policy: Literal["keep", "drop", "nearest"] = "keep",
    source_nan_mask: Annotated[np.ndarray, AfterValidator(_source_nan_mask)]
    | None = None,
) -> Path:
    """Build the CaMa mapping of the dataset's source grid and save it.

    With ``parameter_nc``, rows follow its ``catchment_id`` order.
    ``source_nan_policy="drop"`` removes source cells missing in the first
    frame (or ``source_nan_mask``) while preserving each catchment's row sum;
    ``"nearest"`` additionally repairs catchments left empty with their
    nearest valid source cell.
    """

    space = dataset.space
    if not isinstance(space, GridSpace):
        raise TypeError(f"{type(dataset).__name__} has no source grid to map")
    if source_nan_policy != "keep":
        nan_mask = (
            dataset._first_frame_missing()
            if source_nan_mask is None
            else source_nan_mask
        )
        if nan_mask is None:
            raise ValueError(
                "dataset cannot infer a source NaN mask; pass "
                "source_nan_mask explicitly or use source_nan_policy='keep'"
            )
        if nan_mask.shape != space.shape:
            raise ValueError(
                f"source_nan_mask must have full grid shape {space.shape}, got {nan_mask.shape}"
            )
    mapping = build_cama_mapping(
        space.longitude,
        space.latitude,
        map_dir,
        source_lon_bounds=space.longitude_bounds,
        source_lat_bounds=space.latitude_bounds,
        hires_tag=hires_tag,
        mapinfo_txt=mapinfo_txt,
        lowres_idx_precision=lowres_idx_precision,
        hires_idx_precision=hires_idx_precision,
        map_precision=map_precision,
        parameter_nc=parameter_nc,
        allow_oob_zero=allow_oob_zero,
        producer=f"{type(dataset).__name__}.generate_mapping_table",
    )
    if source_nan_policy != "keep":
        mapping = mapping.with_source_mask(
            np.logical_not(nan_mask),
            empty_row_policy="nearest" if source_nan_policy == "nearest" else "zero",
            preserve_row_sum=True,
        )
        if is_rank_zero():
            logger.info(
                "source_nan_policy=%r removed %d NaN source cells and "
                "repaired %d empty targets",
                source_nan_policy,
                int(nan_mask.sum()),
                mapping.metadata.get("source_mask_repaired_rows", 0),
            )
    mapping.save(out_path)
    if is_rank_zero():
        logger.info(
            "Saved grid mapping to %s: shape=%s, nnz=%d, source=%dx%d",
            out_path,
            mapping.matrix.shape,
            mapping.matrix.nnz,
            space.longitude.size,
            space.latitude.size,
        )
    return out_path

"""Internal checkpoint persistence service for :class:`AbstractModel`."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

import numpy as np
import numpy.ma as ma
import torch
from netCDF4 import Dataset
from pydantic import model_validator

from hydroforge.contracts.events import emit
from hydroforge.contracts.errors import (
    ResourceCleanupError,
    distributed_failure_error,
    failure_description,
)
from hydroforge.data.input import InputProxy
from hydroforge.contracts.validation import HydroForgeModel
from hydroforge.serialization.files import atomic_output_path
from hydroforge.serialization.netcdf import (
    LOGICAL_DTYPE_ATTR,
    _atomic_netcdf_dataset_trusted,
    _create_netcdf_variable_trusted,
    _prepare_netcdf_variable_options_trusted,
    decode_netcdf_logical_array,
    netcdf_dtype_encoding,
    normalize_netcdf_variable_options,
)


@dataclass(frozen=True, slots=True)
class _InputField:
    """One live field required to reconstruct the initialized model."""

    name: str
    module_name: str
    module: Any
    info: Any
    shape: tuple[int, ...]
    numpy_dtype: np.dtype
    coordinate: str | None
    partition_axis: int | None


@dataclass(frozen=True, slots=True)
class _CheckpointPlan:
    """Trusted construction-input layout compiled for the save service."""

    fields: tuple[_InputField, ...]
    layout_signature: tuple[Any, ...]


@dataclass(frozen=True, slots=True)
class _CheckpointSaveStage:
    timestamp: str
    path: Any
    data: dict[str, Any]
    distributed: tuple[str, ...]
    global_fields: tuple[str, ...]
    groups: dict[str, str]


class _CheckpointMergeDeclaration(HydroForgeModel):
    """Complete validated input for one distributed checkpoint merge."""

    output_path: str | Path
    rank_paths: tuple[str | Path, ...]
    variable_group_mapping: Mapping[str, str]
    netcdf_options: Mapping[str, Any]

    @model_validator(mode="after")
    def _validate_merge(self):
        if not self.rank_paths:
            raise ValueError("checkpoint merge requires at least one rank file")
        paths = tuple(Path(path) for path in self.rank_paths)
        normalized = tuple(str(path) for path in paths)
        if len(normalized) != len(set(normalized)):
            raise ValueError("checkpoint merge received duplicate rank files")
        distributed_names = set(self.variable_group_mapping)
        unknown_groups = set(self.variable_group_mapping.values()).difference(
            distributed_names
        )
        if unknown_groups:
            raise ValueError(
                "checkpoint variable groups must name mapped coordinate "
                f"variables: {sorted(unknown_groups)}"
            )
        object.__setattr__(self, "output_path", Path(self.output_path))
        object.__setattr__(self, "rank_paths", paths)
        object.__setattr__(
            self,
            "netcdf_options",
            normalize_netcdf_variable_options(self.netcdf_options),
        )
        return self


def _merge_rank_checkpoints(
    output_path: str | Path,
    rank_paths: tuple[str | Path, ...],
    variable_group_mapping: Mapping[str, str],
    *,
    netcdf_options: Mapping[str, Any],
) -> None:
    """Merge one exact set of rank-local checkpoint files.

    Every mapped variable must occur on every rank. Rank-local files must not
    carry global attributes, so the merged parameter file has none either.
    """
    declaration = _CheckpointMergeDeclaration(
        output_path=output_path,
        rank_paths=rank_paths,
        variable_group_mapping=variable_group_mapping,
        netcdf_options=netcdf_options,
    )
    output_path = declaration.output_path
    rank_paths = declaration.rank_paths
    variable_group_mapping = declaration.variable_group_mapping
    create_options = declaration.netcdf_options
    distributed_names = set(variable_group_mapping)
    offsets: dict[str, int] = {}
    coordinate_groups = set(variable_group_mapping.values())
    coordinate_parts: dict[str, list[np.ndarray]] = {
        name: [] for name in coordinate_groups
    }

    with _atomic_netcdf_dataset_trusted(
        output_path, format="NETCDF4",
    ) as merged_ds:
        for r, rank_path in enumerate(rank_paths):
            with Dataset(rank_path, "r") as rank_ds:
                attrs = {
                    name: rank_ds.getncattr(name) for name in rank_ds.ncattrs()
                }
                if attrs:
                    raise ValueError(
                        f"Rank checkpoint {rank_path!s} must not contain "
                        f"global attributes: {sorted(attrs)}"
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
                    if ma.isMaskedArray(raw_data) and np.any(
                        ma.getmaskarray(raw_data)
                    ):
                        raise ValueError(
                            f"Rank checkpoint {rank_path!s} variable "
                            f"{var_name!r} contains missing values"
                        )
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

                        variable_options = _prepare_netcdf_variable_options_trusted(
                            create_options,
                            dtype=storage_dtype,
                            dimensions=tuple(dims),
                            name=var_name,
                            logical_dtype=logical_dtype,
                        )
                        merged_var = _create_netcdf_variable_trusted(
                            merged_ds,
                            var_name,
                            storage_dtype,
                            tuple(dims),
                            options=variable_options,
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


class CheckpointRuntime:
    """Persist complete inputs for constructing a fresh model."""

    def __init__(self, model: Any) -> None:
        self.model = model
        fields = self._compile_input_fields()
        self.plan = _CheckpointPlan(
            fields=fields,
            layout_signature=tuple(
                (
                    field.name,
                    field.module_name,
                    field.shape,
                    str(field.numpy_dtype),
                    field.coordinate,
                    field.partition_axis,
                )
                for field in fields
            ),
        )

    def _coordinate_phase(
        self,
        error: BaseException | None,
        *,
        phase: str,
        signature: tuple[Any, ...] | None = None,
    ) -> tuple[dict[str, str] | None, ...]:
        """Route every checkpoint collective through the public protocol."""

        if self.model.world_size == 1:
            return (
                None if error is None else failure_description(error),
            )
        return self.model._gather_distributed_failures(
            error,
            phase=phase,
            signature=signature,
        )

    def _coordinate_save_entry(
        self,
        error: BaseException | None,
        stage: _CheckpointSaveStage | None,
    ) -> tuple[tuple[dict[str, str] | None, ...], str | None]:
        """Validate the save declaration and publish one rank-zero nonce."""

        model = self.model
        candidate = uuid4().hex if model.rank == 0 else None
        signature = (
            None
            if stage is None else (
                stage.timestamp,
                self.plan.layout_signature,
                stage.distributed,
                tuple(sorted(stage.groups.items())),
            )
        )
        if model.world_size == 1:
            failures = (
                None if error is None else failure_description(error),
            )
            return failures, None if error is not None else candidate
        failures, payloads = model._exchange_distributed_public_transaction(
            error,
            phase="checkpoint.save.entry",
            signature=signature,
            payload=candidate,
        )
        if any(failure is not None for failure in failures):
            return failures, None
        checkpoint_id = payloads[0]
        if (
            not isinstance(checkpoint_id, str)
            or not checkpoint_id
            or any(value is not None for value in payloads[1:])
        ):
            raise RuntimeError(
                "distributed checkpoint save identity must be generated "
                "exactly once by rank zero"
            )
        return failures, checkpoint_id

    @staticmethod
    def _numpy_dtype(value: Any) -> np.dtype:
        if isinstance(value, torch.Tensor):
            return np.asarray(value.detach().cpu().numpy()).dtype
        return np.asarray(value).dtype

    def _local_input_value(self, name: str) -> Any:
        """Reload one rank-local input whose runtime storage was discarded."""

        source = self.model._input
        group = self.model._partition.variable_groups.get(name)
        if group is None:
            return source[name]
        indices = self.model._partition.rank_indices(group)
        axis = self.model._semantic_plan.input_axes[name]
        selector = (slice(None), indices) if axis == 1 else indices
        return source.get_subset(name, selector)

    def _field_value(self, field: _InputField) -> Any:
        value = getattr(field.module, field.name)
        if value is None:
            value = self._local_input_value(field.name)
        return value

    def _compile_input_fields(self) -> tuple[_InputField, ...]:
        """Compile every current value needed by a fresh model construction."""

        model = self.model
        partition = model._partition
        fields: dict[str, _InputField] = {}
        for name, info in model._input.fields.items():
            tensor = info.tensor
            if tensor is not None:
                if tensor.category not in {"param", "topology", "init_state"}:
                    continue
            elif not info.required and name not in model._input:
                # An absent optional scalar is reconstructed from its declared
                # default and need not be encoded as a NetCDF variable.
                continue
            module = model._modules[info.module_name]
            if name in module.nc_excluded_fields or info.excluded:
                continue
            value = getattr(module, name)
            if value is None:
                if name not in model._input:
                    # A discarded optional tensor that originated from its
                    # declared default is reconstructed from that same default.
                    continue
                value = self._local_input_value(name)
            shape = tuple(np.shape(
                value.detach().cpu().numpy()
                if isinstance(value, torch.Tensor) else value
            ))
            coordinate = partition.field_coordinate(info)
            fields[name] = _InputField(
                name=name,
                module_name=info.module_name,
                module=module,
                info=info,
                shape=shape,
                numpy_dtype=self._numpy_dtype(value),
                coordinate=coordinate,
                partition_axis=(
                    partition.logical_axis(name, info, shape)
                    if coordinate is not None else None
                ),
            )
        return tuple(fields[name] for name in sorted(fields))

    def _stage_save(self) -> _CheckpointSaveStage:
        """Snapshot complete current construction input without publishing."""

        model = self.model
        variable_map = model._namespace.build()
        fields = self.plan.fields
        current_time = model.current_time
        timestamp = (
            current_time.strftime("%Y%m%d_%H%M%S")
            if current_time else "latest"
        )
        name = (
            f"model_state_rank{model.rank}_{timestamp}.nc"
            if model.world_size > 1 else f"model_state_{timestamp}.nc"
        )
        path = model.output_full_dir / name
        data: dict[str, Any] = {}
        distributed: list[str] = []
        global_fields: list[str] = []
        groups: dict[str, str] = {}

        for field in fields:
            if field.coordinate is not None:
                groups[field.name] = field.coordinate
            elif model.rank != 0:
                continue
            value = self._field_value(field)
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy().copy()
            elif isinstance(value, np.ndarray):
                value = np.array(value, order="K", copy=True, subok=False)
            elif isinstance(value, np.generic):
                value = value.copy()
            elif type(value) not in {bool, int, float}:
                raise TypeError(
                    f"checkpoint construction input {field.name!r} has "
                    f"unsupported value type {type(value).__name__!r}"
                )
            data[field.name] = value
            (distributed if field.coordinate is not None else global_fields).append(
                field.name
            )

        for coordinate in sorted(set(groups.values())):
            if coordinate in data:
                continue
            entry = variable_map[coordinate]
            value = getattr(entry.module, entry.field_name)
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy().copy()
            data[coordinate] = value
            groups[coordinate] = coordinate
            distributed.append(coordinate)

        return _CheckpointSaveStage(
            timestamp=timestamp,
            path=path,
            data=data,
            distributed=tuple(distributed),
            global_fields=tuple(global_fields),
            groups=groups,
        )

    def _abort_save(
        self,
        path: Path,
        primary: BaseException,
        *,
        checkpoint_id: str,
    ) -> None:
        """Remove rank-local staging and coordinate the rollback phase."""

        rollback_error: BaseException | None = None
        if self.model.world_size > 1:
            try:
                path.unlink(missing_ok=True)
            except BaseException as error:
                rollback_error = error
        rollback_failures = self._coordinate_phase(
            rollback_error,
            phase="checkpoint.save.rollback",
            signature=(checkpoint_id,),
        )
        if any(failure is not None for failure in rollback_failures):
            rollback_failure = (
                rollback_error
                if rollback_error is not None
                else distributed_failure_error(
                    "distributed checkpoint save rollback",
                    rollback_failures,
                )
            )
            failure = ResourceCleanupError(
                "checkpoint save rollback",
                (primary, rollback_failure),
            )
            raise failure from primary
        raise primary

    def _emit_save_events(
        self,
        *,
        distributed: tuple[str, ...],
        global_fields: tuple[str, ...],
    ) -> None:
        model = self.model
        for event, message, fields in (
            (
                "checkpoint.saved_distributed",
                "Saved distributed state fields", distributed,
            ),
            (
                "checkpoint.saved_global",
                "Saved global state fields", global_fields,
            ),
        ):
            if fields:
                emit(
                    model, "info", event, message, rank=model.rank,
                    fields=tuple(fields),
                )

    def _rollback_single_rank_save(
        self,
        transaction: Any,
        primary: BaseException,
        *,
        checkpoint_id: str,
    ) -> None:
        """Close one staging transaction before reporting save rollback."""

        rollback_error: BaseException | None = None
        try:
            suppressed = transaction.__exit__(
                type(primary), primary, primary.__traceback__,
            )
            if suppressed:
                rollback_error = RuntimeError(
                    "checkpoint staging transaction suppressed its primary error"
                )
        except BaseException as error:
            rollback_error = error
        rollback_failures = self._coordinate_phase(
            rollback_error,
            phase="checkpoint.save.rollback",
            signature=(checkpoint_id,),
        )
        if any(failure is not None for failure in rollback_failures):
            rollback_failure = cast(BaseException, rollback_error)
            failure = ResourceCleanupError(
                "checkpoint save rollback",
                (primary, rollback_failure),
            )
            raise failure from primary
        raise primary

    def _save_single_rank(
        self,
        stage: _CheckpointSaveStage,
        proxy: InputProxy,
        *,
        checkpoint_id: str,
    ) -> InputProxy:
        """Publish one checkpoint only after its pre-commit events succeed."""

        model = self.model
        path = stage.path
        timestamp = stage.timestamp
        transaction = atomic_output_path(path, preserve_suffix=True)
        staging_path: Path | None = None
        write_error: BaseException | None = None
        try:
            staging_path = transaction.__enter__()
            if path.exists():
                emit(
                    model, "warning", "checkpoint.overwrite",
                    "Overwriting existing model state",
                    rank=model.rank, path=path,
                )
            proxy._to_nc(
                staging_path,
                netcdf_options=model.checkpoint_netcdf_options,
            )
        except BaseException as error:
            write_error = error
        write_failures = self._coordinate_phase(
            write_error,
            phase="checkpoint.save.write",
            signature=(checkpoint_id, timestamp),
        )
        if any(failure is not None for failure in write_failures):
            failure = cast(BaseException, write_error)
            if staging_path is None:
                raise failure
            self._rollback_single_rank_save(
                transaction, failure, checkpoint_id=checkpoint_id,
            )

        event_error: BaseException | None = None
        try:
            self._emit_save_events(
                distributed=stage.distributed,
                global_fields=stage.global_fields,
            )
        except BaseException as error:
            event_error = error
        event_failures = self._coordinate_phase(
            event_error,
            phase="checkpoint.save.events.precommit",
            signature=(checkpoint_id, timestamp),
        )
        if any(failure is not None for failure in event_failures):
            self._rollback_single_rank_save(
                transaction,
                cast(BaseException, event_error),
                checkpoint_id=checkpoint_id,
            )

        commit_error: BaseException | None = None
        try:
            transaction.__exit__(None, None, None)
            if not path.is_file():
                raise FileNotFoundError(
                    f"Checkpoint commit point is missing: {path}"
                )
        except BaseException as error:
            commit_error = error
        commit_failures = self._coordinate_phase(
            commit_error,
            phase="checkpoint.save.commit",
            signature=(checkpoint_id, timestamp, str(path)),
        )
        if any(failure is not None for failure in commit_failures):
            failure = cast(BaseException, commit_error)
            model._execution.poison(failure, phase="checkpoint save commit")
            raise failure
        return proxy

    def save(self) -> InputProxy:
        """Persist a checkpoint through rank-synchronous failure phases."""

        model = self.model
        stage = None
        stage_error: BaseException | None = None
        try:
            stage = self._stage_save()
        except BaseException as error:
            stage_error = error
        stage_failures, checkpoint_id = self._coordinate_save_entry(
            stage_error, stage,
        )
        if any(failure is not None for failure in stage_failures):
            if stage_error is not None:
                raise stage_error
            raise distributed_failure_error(
                "distributed checkpoint state snapshot",
                stage_failures,
            )
        stage = cast(_CheckpointSaveStage, stage)
        checkpoint_id = cast(str, checkpoint_id)
        path = stage.path
        timestamp = stage.timestamp
        data = stage.data
        distributed = stage.distributed
        global_fields = stage.global_fields
        groups = stage.groups
        execution = model._execution
        proxy = InputProxy(data=data)
        if model.world_size == 1:
            return self._save_single_rank(
                stage, proxy, checkpoint_id=checkpoint_id,
            )
        write_error: BaseException | None = None
        try:
            if path.exists():
                emit(
                    model, "warning", "checkpoint.overwrite",
                    "Overwriting existing model state",
                    rank=model.rank, path=path,
                )
            proxy._to_nc(
                path,
                netcdf_options=(
                    model.checkpoint_netcdf_options if model.world_size == 1 else {}
                ),
            )
        except BaseException as error:
            write_error = error
        write_failures = self._coordinate_phase(
            write_error,
            phase="checkpoint.save.write",
            signature=(checkpoint_id, timestamp),
        )
        if any(failure is not None for failure in write_failures):
            failure = (
                write_error
                if write_error is not None
                else distributed_failure_error(
                    "distributed checkpoint rank write",
                    write_failures,
                )
            )
            self._abort_save(
                path, failure, checkpoint_id=checkpoint_id,
            )
        event_error: BaseException | None = None
        try:
            self._emit_save_events(
                distributed=distributed,
                global_fields=global_fields,
            )
        except BaseException as error:
            event_error = error
        event_failures = self._coordinate_phase(
            event_error,
            phase="checkpoint.save.events.precommit",
            signature=(checkpoint_id, timestamp),
        )
        if any(failure is not None for failure in event_failures):
            failure = (
                event_error
                if event_error is not None
                else distributed_failure_error(
                    "distributed checkpoint pre-commit event",
                    event_failures,
                )
            )
            self._abort_save(
                path, failure, checkpoint_id=checkpoint_id,
            )

        committed_proxy = proxy
        if model.world_size > 1:
            merge_error: BaseException | None = None
            rank_paths = ()
            merged = model.output_full_dir / f"model_state_{timestamp}.nc"
            if model.rank == 0:
                rank_paths = tuple(
                    model.output_full_dir
                    / f"model_state_rank{rank}_{timestamp}.nc"
                    for rank in range(model.world_size)
                )
                try:
                    _merge_rank_checkpoints(
                        merged, rank_paths, groups,
                        netcdf_options=model.checkpoint_netcdf_options,
                    )
                except BaseException as error:
                    merge_error = error
            merge_failures = self._coordinate_phase(
                merge_error,
                phase="checkpoint.save.merge",
                signature=(checkpoint_id, timestamp, str(merged)),
            )
            if any(failure is not None for failure in merge_failures):
                failure = (
                    merge_error
                    if merge_error is not None
                    else distributed_failure_error(
                        "distributed checkpoint rank merge",
                        merge_failures,
                    )
                )
                self._abort_save(
                    path, failure, checkpoint_id=checkpoint_id,
                )
            commit_error: BaseException | None = None
            if not merged.is_file():
                commit_error = FileNotFoundError(
                    f"Merged checkpoint commit point is missing: {merged}"
                )
            commit_failures = self._coordinate_phase(
                commit_error,
                phase="checkpoint.save.commit",
                signature=(checkpoint_id, timestamp, str(merged)),
            )
            if any(failure is not None for failure in commit_failures):
                failure = (
                    commit_error
                    if commit_error is not None
                    else distributed_failure_error(
                        "distributed checkpoint commit",
                        commit_failures,
                    )
                )
                execution.poison(failure, phase="checkpoint save commit")
                raise failure
            # The atomic merged file is the checkpoint commit point. Rank-file
            # removal is post-commit garbage collection: allowing a partial
            # cleanup failure to turn a published checkpoint into a reported
            # merge failure would make retry impossible once an earlier rank
            # file had already been removed.
            post_commit_errors: list[BaseException] = []
            if model.rank == 0:
                try:
                    emit(
                        model, "info", "checkpoint.merged",
                        "Merged distributed state", rank=0, path=merged,
                    )
                except BaseException as error:
                    post_commit_errors.append(error)
                cleanup_failures = []
                for rank_path in rank_paths:
                    try:
                        rank_path.unlink(missing_ok=True)
                    except BaseException as error:
                        cleanup_failures.append({
                            "path": str(rank_path),
                            **failure_description(error),
                        })
                if cleanup_failures:
                    try:
                        emit(
                            model, "warning", "checkpoint.cleanup_failed",
                            "Merged checkpoint was published but temporary rank "
                            "files could not all be removed",
                            rank=0, failures=tuple(cleanup_failures),
                        )
                    except BaseException as error:
                        post_commit_errors.append(error)
            post_commit_error = (
                None
                if not post_commit_errors
                else post_commit_errors[0]
                if len(post_commit_errors) == 1
                else ResourceCleanupError(
                    "checkpoint post-commit events",
                    post_commit_errors,
                )
            )
            post_commit_failures = self._coordinate_phase(
                post_commit_error,
                phase="checkpoint.save.events.postcommit",
                signature=(checkpoint_id, timestamp),
            )
            if any(
                failure is not None for failure in post_commit_failures
            ):
                error = (
                    post_commit_error
                    if post_commit_error is not None
                    else distributed_failure_error(
                        "distributed checkpoint post-commit event",
                        post_commit_failures,
                    )
                )
                execution.poison(error, phase="checkpoint post-commit event")
                raise error
            # Return the committed, globally merged construction input on
            # every rank. A rank-local staging proxy omits global fields away
            # from rank zero and is therefore not a complete model input.
            reopen_error: BaseException | None = None
            reopened_proxy: InputProxy | None = None
            try:
                reopened_proxy = InputProxy.from_nc(merged, lazy=True)
            except BaseException as error:
                reopen_error = error
            reopen_failures = self._coordinate_phase(
                reopen_error,
                phase="checkpoint.save.reopen",
                signature=(checkpoint_id, timestamp, str(merged)),
            )
            if any(failure is not None for failure in reopen_failures):
                failure = (
                    reopen_error
                    if reopen_error is not None
                    else distributed_failure_error(
                        "distributed checkpoint merged checkpoint reopen",
                        reopen_failures,
                    )
                )
                execution.poison(failure, phase="checkpoint save reopen")
                raise failure
            committed_proxy = cast(InputProxy, reopened_proxy)
        return committed_proxy

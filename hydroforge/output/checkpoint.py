"""Internal checkpoint persistence service for :class:`AbstractModel`."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterator, cast
from uuid import uuid4

import numpy as np
import numpy.ma as ma
import torch
from netCDF4 import Dataset
from pydantic import InstanceOf, PrivateAttr, ValidationInfo, model_validator

from hydroforge.data.distributed import _find_indices_in_trusted
from hydroforge.contracts.events import emit
from hydroforge.contracts.fields import tensor_is_active
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


_CHECKPOINT_FORMAT = "hydroforge.model-state"
_CHECKPOINT_VERSION = 6
_CHECKPOINT_LOAD_CONTEXT = "hydroforge_checkpoint_runtime"


@dataclass(frozen=True, slots=True)
class _StateField:
    name: str
    module_name: str
    module: Any
    info: Any
    tensor: torch.Tensor
    shape: tuple[int, ...]
    numpy_dtype: np.dtype
    coordinate: str | None
    partition_axis: int | None


@dataclass(frozen=True, slots=True)
class _CheckpointCoordinate:
    """Trusted local coordinate identity compiled with the runtime."""

    values: np.ndarray


@dataclass(frozen=True, slots=True)
class _TensorRestore:
    field: _StateField
    array: np.ndarray


@dataclass(frozen=True, slots=True)
class _CheckpointPlan:
    """Complete trusted checkpoint schema compiled after module construction."""

    fields: tuple[_StateField, ...]
    coordinates: Mapping[str, _CheckpointCoordinate]
    manifest: dict[str, Any]
    schema_attrs: dict[str, Any]


@dataclass(frozen=True, slots=True)
class _ValidatedCheckpointPayload:
    """Fully validated external checkpoint content ready for commit."""

    restores: tuple[_TensorRestore, ...]
    checkpoint_id: str


class _CheckpointLoadRequest(HydroForgeModel):
    """One public load request with completely staged external content."""

    proxy: InstanceOf[InputProxy]

    _payload: _ValidatedCheckpointPayload = PrivateAttr()

    @model_validator(mode="after")
    def _validate_checkpoint(self, info: ValidationInfo):
        runtime = (
            info.context.get(_CHECKPOINT_LOAD_CONTEXT)
            if isinstance(info.context, Mapping) else None
        )
        if runtime is None:
            raise ValueError("checkpoint load requires runtime context")
        self._payload = runtime._validate_load_payload(self.proxy)
        return self

    @property
    def payload(self) -> _ValidatedCheckpointPayload:
        return self._payload


@dataclass(frozen=True, slots=True)
class _CheckpointSaveStage:
    timestamp: str
    path: Any
    data: dict[str, Any]
    distributed: tuple[str, ...]
    global_fields: tuple[str, ...]
    groups: dict[str, str]
    attrs: dict[str, Any]


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

    Every mapped variable must occur on every rank.  HydroForge contract
    attributes must agree exactly; rank zero's complete attributes are
    retained in the merged file.  A malformed rank set is rejected before
    it can masquerade as a resumable checkpoint.
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
    contract_attrs: dict[str, Any] | None = None
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
    """Save and restore model state without adding downstream model surface."""

    def __init__(self, model: Any) -> None:
        self.model = model
        fields = self._compile_fields()
        coordinates = self._compile_coordinates(fields)
        manifest = self._manifest(fields)
        encoded = self._canonical_json(manifest)
        self.plan = _CheckpointPlan(
            fields=fields,
            coordinates=coordinates,
            manifest=manifest,
            schema_attrs={
                "hydroforge_checkpoint_format": _CHECKPOINT_FORMAT,
                "hydroforge_checkpoint_version": _CHECKPOINT_VERSION,
                "hydroforge_checkpoint_manifest": encoded,
                "hydroforge_checkpoint_schema": hashlib.sha256(
                    encoded.encode("utf-8")
                ).hexdigest(),
            },
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
                self.plan.schema_attrs["hydroforge_checkpoint_schema"],
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
    def _state_fields(module: Any) -> Iterator[tuple[str, Any]]:
        for field in module.tensor_schema():
            if not tensor_is_active(
                field.tensor, getattr(module, "opened_modules", ()),
            ):
                continue
            if not field.computed and field.tensor.category == "init_state":
                yield field.name, field

    def _compile_fields(self) -> tuple[_StateField, ...]:
        """Compile the exact checkpoint field set from declared model state."""

        model = self.model
        partition = model._partition
        numpy_dtypes = {
            torch.bool: np.dtype("bool"),
            torch.float32: np.dtype("float32"),
            torch.float64: np.dtype("float64"),
            torch.int32: np.dtype("int32"),
            torch.int64: np.dtype("int64"),
        }
        fields: dict[str, _StateField] = {}
        for module_name in model.opened_modules:
            module = model._modules[module_name]
            for field_name, info in self._state_fields(module):
                if field_name in module.nc_excluded_fields or info.excluded:
                    continue
                value = getattr(module, field_name)
                coordinate = partition.field_coordinate(info)
                candidate = _StateField(
                    name=field_name,
                    module_name=module_name,
                    module=module,
                    info=info,
                    tensor=value,
                    shape=tuple(value.shape),
                    numpy_dtype=numpy_dtypes[value.dtype],
                    coordinate=coordinate,
                    partition_axis=(
                        partition.logical_axis(
                            field_name, info, tuple(value.shape),
                        )
                        if coordinate is not None else None
                    ),
                )
                fields[field_name] = candidate
        return tuple(fields[name] for name in sorted(fields))

    def _compile_coordinates(
        self, fields: tuple[_StateField, ...],
    ) -> Mapping[str, _CheckpointCoordinate]:
        """Capture trusted local coordinate identities exactly once."""

        variable_map = self.model._namespace.build()
        coordinates: dict[str, _CheckpointCoordinate] = {}
        for name in sorted({
            field.coordinate for field in fields
            if field.coordinate is not None
        }):
            entry = variable_map[name]
            value = getattr(entry.module, entry.field_name)
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy()
            frozen = np.array(value, copy=True, subok=False)
            frozen.setflags(write=False)
            coordinates[name] = _CheckpointCoordinate(values=frozen)
        return MappingProxyType(coordinates)

    def _manifest(self, fields: tuple[_StateField, ...]) -> dict[str, Any]:
        model = self.model
        entries = []
        for field in fields:
            metadata = field.info.tensor
            entries.append({
                "name": field.name,
                "module": field.module_name,
                "category": metadata.category,
                "computed": bool(field.info.computed),
                "declared_shape": list(metadata.shape),
                "declared_dtype": metadata.dtype,
                "runtime_dtype": str(field.tensor.dtype).removeprefix("torch."),
                "coordinate": field.coordinate,
            })
        return {
            "model": f"{type(model).__module__}.{type(model).__qualname__}",
            "modules": list(model.opened_modules),
            "fields": entries,
        }

    @staticmethod
    def _canonical_json(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _integer_attr(attrs: dict[str, Any], name: str) -> int:
        """Decode one explicitly integral NetCDF attribute at the I/O edge."""

        value = attrs[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer),
        ):
            raise TypeError(f"checkpoint attribute {name!r} must be an integer")
        return int(value)

    def _validate_schema(
        self, proxy: InputProxy,
    ) -> dict[str, Any]:
        attrs = proxy.attrs
        if attrs["hydroforge_checkpoint_format"] != _CHECKPOINT_FORMAT:
            raise ValueError(
                "input is not a versioned HydroForge model-state checkpoint"
            )
        version = self._integer_attr(attrs, "hydroforge_checkpoint_version")
        if version != _CHECKPOINT_VERSION:
            raise ValueError(
                f"unsupported checkpoint version {version}; expected "
                f"{_CHECKPOINT_VERSION}"
            )
        encoded = attrs["hydroforge_checkpoint_manifest"]
        digest = attrs["hydroforge_checkpoint_schema"]
        if not isinstance(encoded, str) or not isinstance(digest, str):
            raise ValueError("checkpoint schema manifest must be text")
        actual_digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        if digest != actual_digest:
            raise ValueError("checkpoint schema manifest digest is invalid")
        manifest = json.loads(encoded)
        expected = self.plan.manifest
        if manifest != expected:
            raise ValueError(
                "checkpoint state schema does not match the initialized model"
            )
        expected_data = {field.name for field in self.plan.fields}
        expected_data.update(
            field.coordinate
            for field in self.plan.fields
            if field.coordinate is not None
        )
        available = set(proxy.keys())
        missing = expected_data.difference(available)
        extra = available.difference(expected_data)
        if missing or extra:
            raise ValueError(
                "checkpoint variables do not match its model-state schema: "
                f"missing={sorted(missing)}, extra={sorted(extra)}"
            )
        return manifest

    @staticmethod
    def _validate_array_dtype(
        field: _StateField, array: np.ndarray,
    ) -> np.ndarray:
        expected = field.numpy_dtype
        if expected == np.dtype("bool") and array.dtype == np.dtype("uint8"):
            if np.any((array != 0) & (array != 1)):
                raise ValueError(
                    f"Boolean checkpoint field {field.name!r} contains values "
                    "other than 0 and 1"
                )
            return array.astype(np.bool_, copy=False)
        if array.dtype != expected:
            raise TypeError(
                f"Dtype mismatch for checkpoint state {field.name!r}: "
                f"expected {expected}, got {array.dtype}"
            )
        return array

    @staticmethod
    def _copy_state(target: torch.Tensor, array: np.ndarray) -> None:
        target.copy_(torch.tensor(array, device=target.device))

    @staticmethod
    def _validate_external_coordinate(
        coordinate: str, value: Any, *, source: str,
    ) -> np.ndarray:
        if np.ma.isMaskedArray(value) and np.any(np.ma.getmaskarray(value)):
            raise ValueError(
                f"Checkpoint coordinate {coordinate!r} contains missing "
                f"IDs in {source}"
            )
        array = np.asarray(value)
        if array.ndim != 1:
            raise ValueError(
                f"Checkpoint coordinate {coordinate!r} in {source} must be "
                "one-dimensional"
            )
        if array.dtype.kind not in "iu":
            raise TypeError(
                f"Checkpoint coordinate {coordinate!r} in {source} must use "
                "an integer dtype"
            )
        if np.unique(array).size != array.size:
            raise ValueError(
                f"Checkpoint coordinate {coordinate!r} in {source} contains "
                "duplicate IDs"
            )
        return array

    def _commit_restores(
        self, restores: Sequence[_TensorRestore],
    ) -> list[_TensorRestore]:
        """Commit tensors transactionally and return their prior values."""

        originals = [
            restore.field.tensor.detach().cpu().numpy().copy()
            for restore in restores
        ]
        touched = 0
        try:
            for restore in restores:
                # Include the currently attempted tensor in rollback: a failed
                # asynchronous/device copy is not guaranteed to be untouched.
                touched += 1
                self._copy_state(restore.field.tensor, restore.array)
        except BaseException as commit_error:
            rollback_errors: list[BaseException] = []
            for restore, original in reversed(tuple(zip(
                restores[:touched], originals[:touched], strict=True,
            ))):
                try:
                    self._copy_state(restore.field.tensor, original)
                except BaseException as rollback_error:
                    rollback_errors.append(rollback_error)
            if rollback_errors:
                error = ResourceCleanupError(
                    "checkpoint restore rollback",
                    (commit_error, *rollback_errors),
                )
                # ``load`` cannot receive the original snapshots when this
                # method raises. Preserve that loss-of-proof on the original
                # aggregate without replacing its public error type or causes.
                error._checkpoint_restore_incomplete = True
                raise error from commit_error
            raise
        return [
            _TensorRestore(restore.field, original)
            for restore, original in zip(restores, originals, strict=True)
        ]

    def _rollback_load(
        self,
        originals: list[_TensorRestore] | None,
        commit_error: BaseException | None,
    ) -> BaseException | None:
        """Restore local pre-load state and report any loss of proof."""

        model = self.model
        rollback_errors: list[BaseException] = []
        if originals is not None:
            try:
                self._commit_restores(originals)
            except BaseException as rollback_error:
                rollback_errors.append(rollback_error)
        try:
            model.checkpoint_state_restored()
        except BaseException as rollback_error:
            rollback_errors.append(rollback_error)
        restore_incomplete = bool(
            commit_error is not None
            and getattr(
                commit_error, "_checkpoint_restore_incomplete", False,
            )
        )
        if not rollback_errors:
            return commit_error if restore_incomplete else None
        failures = (
            (commit_error, *rollback_errors)
            if restore_incomplete else tuple(rollback_errors)
        )
        return ResourceCleanupError("checkpoint load rollback", failures)

    def _stage_save(self) -> _CheckpointSaveStage:
        """Snapshot and validate checkpoint state without publishing a file."""

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
            data[field.name] = field.tensor.detach().cpu().numpy().copy()
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

        attrs = {
            "title": "hydroforge Model State",
            "history": f"Created by hydroforge at {datetime.now().isoformat()}",
            "source": "hydroforge.output.checkpoint.CheckpointRuntime.save",
            **self.plan.schema_attrs,
        }
        return _CheckpointSaveStage(
            timestamp=timestamp,
            path=path,
            data=data,
            distributed=tuple(distributed),
            global_fields=tuple(global_fields),
            groups=groups,
            attrs=attrs,
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
        attrs = stage.attrs
        attrs["hydroforge_checkpoint_id"] = checkpoint_id
        execution = model._execution
        proxy = InputProxy(data=data, attrs=attrs)
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
            # Return the committed, globally merged artifact on every rank.
            # A rank-local staging proxy omits global fields away from rank 0
            # and therefore cannot satisfy this service's own load contract.
            # Once rank 0 broadcasts merge success, absence of the commit
            # point is an I/O failure; silently returning the staging proxy
            # would turn a failed save into a value that cannot be loaded.
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

    def _validate_load_payload(
        self, proxy: InputProxy,
    ) -> _ValidatedCheckpointPayload:
        """Validate a checkpoint completely without mutating live state."""

        fields = self.plan.fields
        self._validate_schema(proxy)
        checkpoint_id = proxy.attrs["hydroforge_checkpoint_id"]
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise ValueError("checkpoint ID must be a non-empty string")
        restores: list[_TensorRestore] = []
        coordinate_indices: dict[str, np.ndarray] = {}
        for field in fields:
            field_name = field.name
            incoming = proxy._get_value_trusted(field_name)
            if isinstance(incoming, torch.Tensor):
                incoming = incoming.detach().cpu().numpy()
            if np.ma.isMaskedArray(incoming) and np.any(
                np.ma.getmaskarray(incoming)
            ):
                raise ValueError(
                    f"Checkpoint state {field_name!r} contains missing values"
                )
            array = np.asarray(incoming)
            coordinate = field.coordinate
            if coordinate is not None:
                indices = coordinate_indices.get(coordinate)
                if indices is None:
                    checkpoint = proxy._get_value_trusted(coordinate)
                    if isinstance(checkpoint, torch.Tensor):
                        checkpoint = checkpoint.detach().cpu().numpy()
                    local = self.plan.coordinates[coordinate].values
                    checkpoint = self._validate_external_coordinate(
                        coordinate, checkpoint, source="checkpoint",
                    )
                    if local.dtype != checkpoint.dtype:
                        raise TypeError(
                            f"Checkpoint coordinate {coordinate!r} dtype "
                            f"{checkpoint.dtype} differs from initialized "
                            f"model dtype {local.dtype}"
                        )
                    indices = _find_indices_in_trusted(
                        local, checkpoint,
                    )
                    if np.any(indices < 0):
                        missing = np.asarray(local)[indices < 0][:5].tolist()
                        raise ValueError(
                            f"Checkpoint coordinate {coordinate!r} is missing "
                            f"local IDs; examples: {missing}"
                        )
                    coordinate_indices[coordinate] = indices
                array = self._slice(
                    field, array, indices,
                )
            elif array.shape != field.shape:
                raise ValueError(
                    f"Shape mismatch for global state {field_name!r}: "
                    f"expected {field.shape}, got {array.shape}"
                )
            if array.shape != field.shape:
                raise ValueError(
                    f"Shape mismatch for {field_name!r} after restore: "
                    f"expected {field.shape}, got {array.shape}"
                )
            array = self._validate_array_dtype(field, array)
            restores.append(_TensorRestore(field, array))

        return _ValidatedCheckpointPayload(tuple(restores), checkpoint_id)

    def load(self, proxy: InputProxy) -> None:
        """Restore one checkpoint as a rank-synchronous transaction."""

        model = self.model
        execution = model._execution
        staged: _ValidatedCheckpointPayload | None = None
        checkpoint_id = None
        validation_error: BaseException | None = None
        try:
            staged = _CheckpointLoadRequest.model_validate(
                {"proxy": proxy},
                context={_CHECKPOINT_LOAD_CONTEXT: self},
            ).payload
            checkpoint_id = staged.checkpoint_id
        except BaseException as error:
            validation_error = error
        validation_failures = self._coordinate_phase(
            validation_error,
            phase="checkpoint.load.entry",
            signature=(checkpoint_id,) if checkpoint_id is not None else None,
        )
        if any(failure is not None for failure in validation_failures):
            if validation_error is not None:
                raise validation_error
            raise distributed_failure_error(
                "distributed checkpoint load validation",
                validation_failures,
            )
        restores = cast(_ValidatedCheckpointPayload, staged).restores
        event_error: BaseException | None = None
        try:
            emit(
                model, "info", "checkpoint.loading", "Loading model state",
                rank=model.rank,
            )
        except BaseException as error:
            event_error = error
        event_failures = self._coordinate_phase(
            event_error,
            phase="checkpoint.load.events.precommit",
            signature=(checkpoint_id,),
        )
        if any(failure is not None for failure in event_failures):
            if event_error is not None:
                raise event_error
            raise distributed_failure_error(
                "distributed checkpoint pre-load event",
                event_failures,
            )

        # Commit only after every rank validated every field, coordinate and
        # tensor. Copies preserve tensor identities, so existing compiled
        # bindings remain valid. Retain old state until every rank reports a
        # successful commit.
        original_tensors: list[_TensorRestore] | None = None
        commit_error: BaseException | None = None
        try:
            original_tensors = self._commit_restores(restores)
            model.checkpoint_state_restored()
        except BaseException as error:
            commit_error = error

        try:
            commit_failures = self._coordinate_phase(
                commit_error,
                phase="checkpoint.load.commit",
                signature=(checkpoint_id,),
            )
        except BaseException as coordination_error:
            rollback_error = self._rollback_load(
                original_tensors, commit_error,
            )
            failures: list[BaseException] = []
            if commit_error is not None:
                failures.append(commit_error)
            failures.append(coordination_error)
            if (
                rollback_error is not None
                and rollback_error is not commit_error
            ):
                failures.append(rollback_error)
            failure = (
                coordination_error
                if len(failures) == 1 else ResourceCleanupError(
                    "checkpoint load commit coordination", failures,
                )
            )
            phase = (
                "checkpoint load commit coordination"
                if rollback_error is None else "checkpoint load rollback"
            )
            execution.poison(failure, phase=phase)
            if failure is coordination_error:
                raise
            raise failure from coordination_error
        if any(failure is not None for failure in commit_failures):
            rollback_error = self._rollback_load(
                original_tensors, commit_error,
            )
            try:
                rollback_failures = self._coordinate_phase(
                    rollback_error,
                    phase="checkpoint.load.rollback",
                    signature=(checkpoint_id,),
                )
            except BaseException as coordination_error:
                failures = []
                if commit_error is not None:
                    failures.append(commit_error)
                else:
                    failures.append(distributed_failure_error(
                        "distributed checkpoint load commit",
                        commit_failures,
                    ))
                if rollback_error is not None:
                    failures.append(rollback_error)
                failures.append(coordination_error)
                failure = ResourceCleanupError(
                    "checkpoint load rollback coordination", failures,
                )
                execution.poison(
                    failure, phase="checkpoint load rollback coordination",
                )
                raise failure from coordination_error
            if any(failure is not None for failure in rollback_failures):
                if rollback_error is None:
                    rollback_error = distributed_failure_error(
                        "distributed checkpoint rollback",
                        rollback_failures,
                    )
                execution.poison(
                    rollback_error, phase="checkpoint load rollback",
                )
                if rollback_error is commit_error:
                    raise rollback_error
                raise rollback_error from commit_error
            if commit_error is not None:
                raise commit_error
            raise distributed_failure_error(
                "distributed checkpoint load commit",
                commit_failures,
            )
        event_error = None
        try:
            emit(
                model, "info", "checkpoint.loaded", "Loaded model state",
                rank=model.rank, variables=len(restores),
            )
        except BaseException as error:
            event_error = error
        try:
            event_failures = self._coordinate_phase(
                event_error,
                phase="checkpoint.load.events.postcommit",
                signature=(checkpoint_id,),
            )
        except BaseException as coordination_error:
            failure = (
                coordination_error
                if event_error is None else ResourceCleanupError(
                    "checkpoint post-load event coordination",
                    (event_error, coordination_error),
                )
            )
            execution.poison(
                failure, phase="checkpoint post-load event coordination",
            )
            if failure is coordination_error:
                raise
            raise failure from coordination_error
        if any(failure is not None for failure in event_failures):
            if event_error is None:
                event_error = distributed_failure_error(
                    "distributed checkpoint post-load event",
                    event_failures,
                )
            if event_error is None:
                raise RuntimeError("checkpoint post-load event failed")
            execution.poison(event_error, phase="checkpoint post-load event")
            raise event_error

    def _slice(
        self,
        field: _StateField,
        array: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        axis = cast(int, field.partition_axis)
        slicer = [slice(None)] * array.ndim
        slicer[axis] = indices
        try:
            return array[tuple(slicer)]
        except IndexError as exc:
            raise ValueError(
                f"Cannot shard checkpoint field {field.name!r} with shape "
                f"{array.shape} on logical axis {axis}"
            ) from exc

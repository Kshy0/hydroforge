"""Internal checkpoint persistence service for :class:`AbstractModel`."""

from __future__ import annotations

from datetime import datetime
from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterator
from uuid import uuid4

import numpy as np
import torch
import torch.distributed as dist

from hydroforge.data.distributed import find_indices_in
from hydroforge.contracts.events import emit
from hydroforge.contracts import ResourceCleanupError
from hydroforge.contracts.temporal import timedelta_microseconds
from hydroforge.data.input import InputProxy


_STATE_CATEGORIES = frozenset({"init_state", "state", "shared_state"})
_CHECKPOINT_FORMAT = "hydroforge.model-state"
_CHECKPOINT_VERSION = 4


@dataclass(frozen=True, slots=True)
class _StateField:
    name: str
    module_name: str
    module: Any
    info: Any
    tensor: torch.Tensor
    coordinate: str | None


@dataclass(frozen=True, slots=True)
class _TensorRestore:
    field: _StateField
    array: np.ndarray


@dataclass(frozen=True, slots=True)
class _CheckpointSaveStage:
    timestamp: str
    path: Any
    data: dict[str, Any]
    distributed: tuple[str, ...]
    global_fields: tuple[str, ...]
    groups: dict[str, str]
    attrs: dict[str, Any]
    temporal_state: Any


class CheckpointRuntime:
    """Save and restore model state without adding downstream model surface."""

    def __init__(self, model: Any) -> None:
        self.model = model

    @staticmethod
    def _error_description(error: BaseException) -> dict[str, str]:
        return {
            "type": f"{type(error).__module__}.{type(error).__qualname__}",
            "message": str(error),
        }

    @staticmethod
    def _raise_remote_failure(phase: str, failures: list[Any]) -> None:
        failed = [
            (rank, failure) for rank, failure in enumerate(failures)
            if failure is not None
        ]
        details = "; ".join(
            f"rank {rank}: {failure['type']}: {failure['message']}"
            for rank, failure in failed
        )
        raise RuntimeError(f"distributed checkpoint {phase} failed: {details}")

    def _gather_failures(
        self, error: BaseException | None,
    ) -> list[Any]:
        """Publish one checkpoint phase result to every rank."""

        model = self.model
        local = None if error is None else self._error_description(error)
        if model.world_size == 1:
            return [local]
        failures: list[Any] = [None] * model.world_size
        dist.all_gather_object(failures, local)
        return failures

    @staticmethod
    def _identity_result(
        value: Any, *, rank: int, phase: str,
    ) -> tuple[Any, str | None]:
        if not isinstance(value, dict) or set(value) != {
            "error", "checkpoint_id",
        }:
            raise RuntimeError(
                f"distributed checkpoint {phase} rank {rank} returned an "
                "invalid identity handshake"
            )
        error = value["error"]
        checkpoint_id = value["checkpoint_id"]
        if error is not None and not (
            isinstance(error, dict)
            and set(error) == {"type", "message"}
            and all(isinstance(item, str) for item in error.values())
        ):
            raise RuntimeError(
                f"distributed checkpoint {phase} rank {rank} returned an "
                "invalid failure description"
            )
        if checkpoint_id is not None and (
            not isinstance(checkpoint_id, str) or not checkpoint_id
        ):
            raise RuntimeError(
                f"distributed checkpoint {phase} rank {rank} returned an "
                "invalid checkpoint ID"
            )
        return error, checkpoint_id

    def _synchronize_save_identity(
        self, error: BaseException | None,
    ) -> tuple[list[Any], str | None]:
        """Combine snapshot failure propagation with one rank-zero nonce."""

        model = self.model
        candidate = uuid4().hex if model.rank == 0 else None
        local = {
            "error": None if error is None else self._error_description(error),
            "checkpoint_id": candidate,
        }
        if model.world_size == 1:
            observed = [local]
        else:
            observed = [None] * model.world_size
            dist.all_gather_object(observed, local)
        decoded = tuple(
            self._identity_result(value, rank=rank, phase="save snapshot")
            for rank, value in enumerate(observed)
        )
        failures = [item[0] for item in decoded]
        if any(failure is not None for failure in failures):
            return failures, None
        checkpoint_ids = tuple(item[1] for item in decoded)
        if (
            checkpoint_ids[0] is None
            or any(value is not None for value in checkpoint_ids[1:])
        ):
            raise RuntimeError(
                "distributed checkpoint save identity must be generated "
                "exactly once by rank zero"
            )
        return failures, checkpoint_ids[0]

    def _synchronize_load_identity(
        self, error: BaseException | None, checkpoint_id: str | None,
    ) -> list[Any]:
        """Combine validation failure propagation with identity agreement."""

        model = self.model
        local = {
            "error": None if error is None else self._error_description(error),
            "checkpoint_id": checkpoint_id,
        }
        if model.world_size == 1:
            observed = [local]
        else:
            observed = [None] * model.world_size
            dist.all_gather_object(observed, local)
        decoded = tuple(
            self._identity_result(value, rank=rank, phase="load validation")
            for rank, value in enumerate(observed)
        )
        failures = [item[0] for item in decoded]
        if any(failure is not None for failure in failures):
            return failures
        checkpoint_ids = tuple(item[1] for item in decoded)
        if checkpoint_ids[0] is None or len(set(checkpoint_ids)) != 1:
            raise ValueError(
                "distributed ranks are loading different checkpoint IDs: "
                f"{checkpoint_ids}"
            )
        return failures

    @staticmethod
    def _state_fields(module: Any) -> Iterator[tuple[str, Any]]:
        for field in module.tensor_schema():
            if any(
                dependency not in module.opened_modules
                for dependency in field.tensor.depends_on
            ):
                continue
            category = field.tensor.category
            if (
                not field.computed and category in _STATE_CATEGORIES
                or field.computed and category in {"state", "shared_state"}
            ):
                yield field.name, field

    def _fields(self) -> tuple[_StateField, ...]:
        """Compile the exact checkpoint field set from declared model state."""

        model = self.model
        partition = model._partition
        fields: dict[str, _StateField] = {}
        for module_name in model.opened_modules:
            module = model._modules[module_name]
            for field_name, info in self._state_fields(module):
                if field_name in module.nc_excluded_fields or info.excluded:
                    continue
                value = getattr(module, field_name)
                if not isinstance(value, torch.Tensor):
                    raise TypeError(
                        "checkpoint state must be an initialized torch.Tensor: "
                        f"{module_name}.{field_name} is {type(value).__name__}"
                    )
                candidate = _StateField(
                    field_name, module_name, module, info, value,
                    partition.field_coordinate(info),
                )
                previous = fields.get(field_name)
                if previous is not None:
                    raise ValueError(
                        f"checkpoint state name {field_name!r} is declared by "
                        f"both {previous.module_name!r} and {module_name!r}; "
                        "state ownership must be unique"
                    )
                fields[field_name] = candidate
        return tuple(fields[name] for name in sorted(fields))

    def _manifest(self, fields: tuple[_StateField, ...]) -> dict[str, Any]:
        model = self.model
        controller = model._execution.step.controller
        statistics = (
            {"mode": "plan", "fingerprint": controller.fingerprint}
            if controller is not None else {
                "mode": "implicit",
                "calendar": model.calendar,
                "inner_microseconds": (
                    None if model.statistics_interval is None
                    else timedelta_microseconds(
                        model.statistics_interval,
                        label="statistics_interval",
                    )
                ),
                "outer_microseconds": (
                    None if model.statistics_outer_interval is None
                    else timedelta_microseconds(
                        model.statistics_outer_interval,
                        label="statistics_outer_interval",
                    )
                ),
            }
        )
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
            "simulation_schedule": (
                None if getattr(model, "simulation_schedule", None) is None
                else model.simulation_schedule.fingerprint
            ),
            "statistics": statistics,
            "fields": entries,
        }

    @staticmethod
    def _canonical_json(value: Any) -> str:
        return json.dumps(value, sort_keys=True, separators=(",", ":"))

    @staticmethod
    def _integer_attr(attrs: dict[str, Any], name: str) -> int:
        """Decode one explicitly integral NetCDF attribute at the I/O edge."""

        if name not in attrs:
            raise ValueError(f"checkpoint is missing integer attribute {name!r}")
        value = attrs[name]
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value, (int, np.integer),
        ):
            raise TypeError(f"checkpoint attribute {name!r} must be an integer")
        return int(value)

    def _schema_attrs(self, fields: tuple[_StateField, ...]) -> dict[str, Any]:
        manifest = self._manifest(fields)
        encoded = self._canonical_json(manifest)
        return {
            "hydroforge_checkpoint_format": _CHECKPOINT_FORMAT,
            "hydroforge_checkpoint_version": _CHECKPOINT_VERSION,
            "hydroforge_checkpoint_manifest": encoded,
            "hydroforge_checkpoint_schema": hashlib.sha256(
                encoded.encode("utf-8")
            ).hexdigest(),
        }

    def _validate_schema(
        self, proxy: InputProxy, fields: tuple[_StateField, ...],
    ) -> dict[str, Any]:
        attrs = proxy.attrs
        if attrs.get("hydroforge_checkpoint_format") != _CHECKPOINT_FORMAT:
            raise ValueError(
                "input is not a versioned HydroForge model-state checkpoint"
            )
        version = self._integer_attr(attrs, "hydroforge_checkpoint_version")
        if version != _CHECKPOINT_VERSION:
            raise ValueError(
                f"unsupported checkpoint version {version}; expected "
                f"{_CHECKPOINT_VERSION}"
            )
        encoded = attrs.get("hydroforge_checkpoint_manifest")
        digest = attrs.get("hydroforge_checkpoint_schema")
        if not isinstance(encoded, str) or not isinstance(digest, str):
            raise ValueError("checkpoint is missing its schema manifest")
        actual_digest = hashlib.sha256(encoded.encode("utf-8")).hexdigest()
        if digest != actual_digest:
            raise ValueError("checkpoint schema manifest digest is invalid")
        try:
            manifest = json.loads(encoded)
        except json.JSONDecodeError as error:
            raise ValueError("checkpoint schema manifest is not valid JSON") from error
        expected = self._manifest(fields)
        if manifest != expected:
            raise ValueError(
                "checkpoint state schema does not match the initialized model"
            )
        expected_data = {field.name for field in fields}
        expected_data.update(
            field.coordinate for field in fields if field.coordinate is not None
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
        field_name: str, array: np.ndarray, current: torch.Tensor,
    ) -> np.ndarray:
        expected_by_dtype = {
            torch.bool: np.dtype("bool"),
            torch.float32: np.dtype("float32"),
            torch.float64: np.dtype("float64"),
            torch.int32: np.dtype("int32"),
            torch.int64: np.dtype("int64"),
        }
        try:
            expected = expected_by_dtype[current.dtype]
        except KeyError as error:
            raise TypeError(
                f"Checkpoint field {field_name!r} has unsupported runtime "
                f"dtype {current.dtype}"
            ) from error
        if current.dtype is torch.bool and array.dtype == np.dtype("uint8"):
            if np.any((array != 0) & (array != 1)):
                raise ValueError(
                    f"Boolean checkpoint field {field_name!r} contains values "
                    "other than 0 and 1"
                )
            return array.astype(np.bool_, copy=False)
        if array.dtype != expected:
            raise TypeError(
                f"Dtype mismatch for checkpoint state {field_name!r}: "
                f"expected {expected}, got {array.dtype}"
            )
        return array

    @staticmethod
    def _copy_state(target: torch.Tensor, array: np.ndarray) -> None:
        target.copy_(torch.as_tensor(array, device=target.device))

    @staticmethod
    def _validated_coordinate(
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
        self, restores: list[_TensorRestore],
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
                raise error from commit_error
            raise
        return [
            _TensorRestore(restore.field, original)
            for restore, original in zip(restores, originals, strict=True)
        ]

    def _stage_save(self, current_time: Any) -> _CheckpointSaveStage:
        """Snapshot and validate checkpoint state without publishing a file."""

        model = self.model
        variable_map = model._namespace.build()
        fields = self._fields()
        if model.num_trials is not None:
            raise ValueError(
                "Checkpoint save currently requires a non-ensemble model"
            )
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
            module, attribute, _ = variable_map[coordinate]
            value = getattr(module, attribute)
            if isinstance(value, torch.Tensor):
                value = value.detach().cpu().numpy().copy()
            data[coordinate] = value
            groups[coordinate] = coordinate
            distributed.append(coordinate)

        attrs = {
            "title": "hydroforge Model State",
            "history": f"Created by hydroforge at {datetime.now().isoformat()}",
            "source": "hydroforge.output.checkpoint.CheckpointRuntime.save",
            **self._schema_attrs(fields),
        }
        execution = model._execution
        step = execution.step
        attrs["hydroforge_managed_step_state"] = self._canonical_json(
            execution.checkpoint_step_state()
        )
        attrs["hydroforge_parameter_plan_state"] = self._canonical_json(
            model._parameters.checkpoint_state()
        )
        attrs["hydroforge_statistics_control"] = step.statistics_control
        attrs["hydroforge_statistics_boundary"] = "closed"
        temporal_state = step.persisted_statistics_state()
        aggregator = execution.statistics.aggregator
        has_outer = bool(
            aggregator is not None
            and any(aggregator._output_is_outer.values())
        )
        inner_open, outer_open = step.open_statistics_windows(
            has_outer=has_outer,
        )
        if aggregator is not None and (inner_open or outer_open):
            raise RuntimeError(
                "checkpointing inside an unfinished statistics window is "
                "not supported; save at an inner/outer window boundary"
            )
        return _CheckpointSaveStage(
            timestamp=timestamp,
            path=path,
            data=data,
            distributed=tuple(distributed),
            global_fields=tuple(global_fields),
            groups=groups,
            attrs=attrs,
            temporal_state=temporal_state,
        )

    def save(self, current_time: Any) -> InputProxy:
        """Persist a checkpoint through rank-synchronous failure phases."""

        model = self.model
        stage = None
        stage_error: BaseException | None = None
        try:
            stage = self._stage_save(current_time)
        except BaseException as error:
            stage_error = error
        stage_failures, checkpoint_id = self._synchronize_save_identity(
            stage_error,
        )
        if any(failure is not None for failure in stage_failures):
            if stage_error is not None:
                raise stage_error
            self._raise_remote_failure("state snapshot", stage_failures)
        if stage is None:
            raise RuntimeError("checkpoint state snapshot produced no plan")
        if checkpoint_id is None:
            raise RuntimeError("checkpoint save produced no checkpoint ID")
        path = stage.path
        timestamp = stage.timestamp
        data = stage.data
        distributed = stage.distributed
        global_fields = stage.global_fields
        groups = stage.groups
        attrs = stage.attrs
        attrs["hydroforge_checkpoint_id"] = checkpoint_id
        temporal_state = stage.temporal_state
        execution = model._execution
        durability_error: BaseException | None = None
        try:
            execution.statistics.ensure_output_durable(current_time)
        except BaseException as error:
            durability_error = error
        durability_failures = self._gather_failures(durability_error)
        if any(failure is not None for failure in durability_failures):
            if durability_error is not None:
                execution.poison(
                    durability_error,
                    phase="statistics output durability",
                )
                raise durability_error
            try:
                self._raise_remote_failure(
                    "statistics durability", durability_failures,
                )
            except RuntimeError as error:
                execution.poison(
                    error, phase="statistics output durability",
                )
                raise
        if temporal_state is not None:
            attrs.update({
                "hydroforge_statistics_plan": temporal_state["fingerprint"],
                "hydroforge_statistics_last_step": temporal_state["last_step_index"],
                "hydroforge_statistics_output_active": temporal_state["output_active"],
                "hydroforge_statistics_inner_open": temporal_state["inner_open"],
                "hydroforge_statistics_outer_open": temporal_state["outer_open"],
            })
        proxy = InputProxy(data, attrs=attrs)
        write_error: BaseException | None = None
        try:
            if path.exists():
                emit(
                    model, "warning", "checkpoint.overwrite",
                    "Overwriting existing model state",
                    rank=model.rank, path=path,
                )
            proxy.to_nc(
                path,
                netcdf_options=(
                    model.checkpoint_netcdf_options if model.world_size == 1 else {}
                ),
            )
        except BaseException as error:
            write_error = error
        write_failures = self._gather_failures(write_error)
        if any(failure is not None for failure in write_failures):
            if write_error is not None:
                raise write_error
            self._raise_remote_failure("rank write", write_failures)
        event_error: BaseException | None = None
        try:
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
        except BaseException as error:
            event_error = error
        event_failures = self._gather_failures(event_error)
        if any(failure is not None for failure in event_failures):
            if event_error is not None:
                raise event_error
            self._raise_remote_failure("pre-commit event", event_failures)

        if model.world_size > 1:
            merge_error: BaseException | None = None
            rank_paths = ()
            if model.rank == 0:
                merged = model.output_full_dir / f"model_state_{timestamp}.nc"
                rank_paths = tuple(
                    model.output_full_dir
                    / f"model_state_rank{rank}_{timestamp}.nc"
                    for rank in range(model.world_size)
                )
                try:
                    InputProxy.merge(
                        merged, rank_paths, groups,
                        netcdf_options=model.checkpoint_netcdf_options,
                    )
                    emit(
                        model, "info", "checkpoint.merged",
                        "Merged distributed state", rank=0, path=merged,
                    )
                except BaseException as error:
                    merge_error = error
            merge_failure = [
                None if merge_error is None
                else self._error_description(merge_error)
            ]
            dist.broadcast_object_list(merge_failure, src=0)
            if merge_failure[0] is not None:
                if merge_error is not None:
                    raise merge_error
                self._raise_remote_failure("rank merge", [merge_failure[0]])
            # The atomic merged file is the checkpoint commit point. Rank-file
            # removal is post-commit garbage collection: allowing a partial
            # cleanup failure to turn a published checkpoint into a reported
            # merge failure would make retry impossible once an earlier rank
            # file had already been removed.
            post_commit_error: BaseException | None = None
            if model.rank == 0:
                cleanup_failures = []
                for rank_path in rank_paths:
                    try:
                        rank_path.unlink(missing_ok=True)
                    except BaseException as error:
                        cleanup_failures.append({
                            "path": str(rank_path),
                            **self._error_description(error),
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
                        post_commit_error = error
            post_commit_failure = [
                None if post_commit_error is None
                else self._error_description(post_commit_error)
            ]
            dist.broadcast_object_list(post_commit_failure, src=0)
            if post_commit_failure[0] is not None:
                if post_commit_error is None:
                    error = RuntimeError(
                        "distributed checkpoint post-commit event failed: "
                        f"rank 0: {post_commit_failure[0]['type']}: "
                        f"{post_commit_failure[0]['message']}"
                    )
                else:
                    error = post_commit_error
                execution.poison(error, phase="checkpoint post-commit event")
                raise error
        return proxy

    def _stage_load(
        self, proxy: InputProxy,
    ) -> tuple[list[_TensorRestore], Any, Any, Any, str]:
        """Validate a checkpoint completely without mutating live state."""

        model = self.model
        partition = model._partition
        variable_map = model._namespace.build()
        fields = self._fields()
        self._validate_schema(proxy, fields)
        checkpoint_id = proxy.attrs.get("hydroforge_checkpoint_id")
        if not isinstance(checkpoint_id, str) or not checkpoint_id:
            raise ValueError("checkpoint is missing its unique checkpoint ID")
        emit(
            model, "info", "checkpoint.loading", "Loading model state",
            rank=model.rank,
        )
        execution = model._execution
        step = execution.step
        encoded_step_state = proxy.attrs.get("hydroforge_managed_step_state")
        if not isinstance(encoded_step_state, str):
            raise ValueError("checkpoint is missing managed-step runtime state")
        try:
            checkpoint_step_state = json.loads(encoded_step_state)
        except json.JSONDecodeError as error:
            raise ValueError(
                "checkpoint managed-step runtime state is not valid JSON"
            ) from error
        staged_step_state = execution.validate_checkpoint_step_state(
            checkpoint_step_state,
        )
        encoded_parameter_state = proxy.attrs.get(
            "hydroforge_parameter_plan_state"
        )
        if not isinstance(encoded_parameter_state, str):
            raise ValueError("checkpoint is missing parameter-plan runtime state")
        try:
            checkpoint_parameter_state = json.loads(encoded_parameter_state)
        except json.JSONDecodeError as error:
            raise ValueError(
                "checkpoint parameter-plan runtime state is not valid JSON"
            ) from error
        staged_parameter_state = model._parameters.validate_checkpoint_state(
            checkpoint_parameter_state,
        )
        temporal_state = None
        expected_control = step.statistics_control
        if proxy.attrs.get("hydroforge_statistics_control") != expected_control:
            raise ValueError(
                "checkpoint statistics control does not match the initialized "
                f"model: expected {expected_control!r}"
            )
        if proxy.attrs.get("hydroforge_statistics_boundary") != "closed":
            raise ValueError(
                "checkpoint does not declare a closed statistics boundary"
            )
        if expected_control == "plan":
            required = {
                "hydroforge_statistics_plan",
                "hydroforge_statistics_last_step",
                "hydroforge_statistics_output_active",
                "hydroforge_statistics_inner_open",
                "hydroforge_statistics_outer_open",
            }
            missing = required.difference(proxy.attrs)
            if missing:
                raise ValueError(
                    "checkpoint is missing statistics plan state: "
                    f"{sorted(missing)}"
                )
            temporal_state = {
                "fingerprint": proxy.attrs["hydroforge_statistics_plan"],
                "last_step_index": self._integer_attr(
                    proxy.attrs, "hydroforge_statistics_last_step",
                ),
                "output_active": self._integer_attr(
                    proxy.attrs, "hydroforge_statistics_output_active",
                ),
                "inner_open": self._integer_attr(
                    proxy.attrs, "hydroforge_statistics_inner_open",
                ),
                "outer_open": self._integer_attr(
                    proxy.attrs, "hydroforge_statistics_outer_open",
                ),
            }
            step.validate_persisted_statistics_state(temporal_state)
            aggregator = execution.statistics.aggregator
            has_outer = bool(
                aggregator is not None
                and any(aggregator._output_is_outer.values())
            )
            if aggregator is not None and (
                temporal_state["inner_open"] == 1
                or temporal_state["outer_open"] == 1 and has_outer
            ):
                raise RuntimeError(
                    "checkpoint contains an unfinished statistics window, "
                    "whose accumulators are not checkpoint state"
                )
        else:
            unexpected = {
                name for name in proxy.attrs
                if name.startswith("hydroforge_statistics_")
                and name not in {
                    "hydroforge_statistics_control",
                    "hydroforge_statistics_boundary",
                }
            }
            if unexpected:
                raise ValueError(
                    "implicit statistics checkpoint contains plan cursor "
                    f"attributes: {sorted(unexpected)}"
                )
        restores: list[_TensorRestore] = []
        coordinate_indices: dict[str, np.ndarray] = {}
        for field in fields:
            field_name = field.name
            info = field.info
            incoming = proxy[field_name]
            current = field.tensor
            if isinstance(incoming, torch.Tensor):
                incoming = incoming.detach().cpu().numpy()
            array = np.asarray(incoming)
            coordinate = field.coordinate
            if coordinate is not None and coordinate in proxy:
                indices = coordinate_indices.get(coordinate)
                if indices is None:
                    coord_module, coord_attribute, _ = variable_map[coordinate]
                    local = getattr(coord_module, coord_attribute)
                    if isinstance(local, torch.Tensor):
                        local = local.detach().cpu().numpy()
                    checkpoint = proxy[coordinate]
                    if isinstance(checkpoint, torch.Tensor):
                        checkpoint = checkpoint.detach().cpu().numpy()
                    local = self._validated_coordinate(
                        coordinate, local, source="initialized model",
                    )
                    checkpoint = self._validated_coordinate(
                        coordinate, checkpoint, source="checkpoint",
                    )
                    if local.dtype != checkpoint.dtype:
                        raise TypeError(
                            f"Checkpoint coordinate {coordinate!r} dtype "
                            f"{checkpoint.dtype} differs from initialized "
                            f"model dtype {local.dtype}"
                        )
                    indices = find_indices_in(
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
                    field_name, info, array, indices,
                )
            elif array.shape != tuple(current.shape):
                if coordinate is None:
                    raise ValueError(
                        f"Shape mismatch for global state {field_name!r}: "
                        f"expected {tuple(current.shape)}, got {array.shape}"
                    )
                array = self._slice(
                    field_name, info, array,
                    partition.rank_indices(coordinate),
                )
            if array.shape != tuple(current.shape):
                raise ValueError(
                    f"Shape mismatch for {field_name!r} after restore: "
                    f"expected {tuple(current.shape)}, got {array.shape}"
                )
            array = self._validate_array_dtype(field_name, array, current)
            restores.append(_TensorRestore(field, array))

        return (
            restores, temporal_state, staged_step_state,
            staged_parameter_state, checkpoint_id,
        )

    def load(self, proxy: InputProxy) -> None:
        """Restore one checkpoint as a rank-synchronous transaction."""

        model = self.model
        execution = model._execution
        step = execution.step
        staged = None
        checkpoint_id = None
        validation_error: BaseException | None = None
        try:
            staged = self._stage_load(proxy)
            checkpoint_id = staged[-1]
        except BaseException as error:
            validation_error = error
        validation_failures = self._synchronize_load_identity(
            validation_error, checkpoint_id,
        )
        if any(failure is not None for failure in validation_failures):
            if validation_error is not None:
                raise validation_error
            self._raise_remote_failure(
                "load validation", validation_failures,
            )
        if staged is None:
            raise RuntimeError("checkpoint load validation produced no plan")
        (
            restores, temporal_state, staged_step_state,
            staged_parameter_state, _checkpoint_id,
        ) = staged

        # Commit only after every rank validated every field, coordinate and
        # temporal cursor. Copies preserve tensor identities, so existing
        # compiled bindings remain valid. Retain all old state until every
        # rank reports a successful commit.
        old_temporal_state = step.persisted_statistics_state()
        old_step_state = execution.validate_checkpoint_step_state(
            execution.checkpoint_step_state(),
        )
        old_parameter_state = model._parameters.validate_checkpoint_state(
            model._parameters.checkpoint_state(),
        )
        original_tensors: list[_TensorRestore] | None = None
        commit_error: BaseException | None = None
        try:
            original_tensors = self._commit_restores(restores)
            step.restore_persisted_statistics_state(temporal_state)
            execution.restore_checkpoint_step_state(staged_step_state)
            model._parameters.restore_checkpoint_state(staged_parameter_state)
        except BaseException as error:
            commit_error = error

        commit_failures = self._gather_failures(commit_error)
        if any(failure is not None for failure in commit_failures):
            rollback_errors: list[BaseException] = []
            rollbacks = (
                lambda: model._parameters.restore_checkpoint_state(
                    old_parameter_state,
                ),
                lambda: execution.restore_checkpoint_step_state(old_step_state),
                lambda: step.restore_persisted_statistics_state(
                    old_temporal_state,
                ),
                lambda: (
                    None if original_tensors is None
                    else self._commit_restores(original_tensors)
                ),
            )
            for rollback in rollbacks:
                try:
                    rollback()
                except BaseException as rollback_error:
                    rollback_errors.append(rollback_error)
            rollback_error: BaseException | None = None
            if rollback_errors:
                rollback_error = ResourceCleanupError(
                    "checkpoint load rollback",
                    tuple(rollback_errors),
                )
            rollback_failures = self._gather_failures(rollback_error)
            if any(failure is not None for failure in rollback_failures):
                if rollback_error is None:
                    rollback_error = RuntimeError(
                        "distributed checkpoint rollback failed on peer "
                        "rank(s): "
                        + "; ".join(
                            f"rank {rank}: {failure['type']}: "
                            f"{failure['message']}"
                            for rank, failure in enumerate(rollback_failures)
                            if failure is not None
                        )
                    )
                execution.poison(
                    rollback_error, phase="checkpoint load rollback",
                )
                raise rollback_error from commit_error
            if commit_error is not None:
                raise commit_error
            self._raise_remote_failure("load commit", commit_failures)
        event_error: BaseException | None = None
        try:
            emit(
                model, "info", "checkpoint.loaded", "Loaded model state",
                rank=model.rank, variables=len(restores),
            )
        except BaseException as error:
            event_error = error
        event_failures = self._gather_failures(event_error)
        if any(failure is not None for failure in event_failures):
            if event_error is None:
                try:
                    self._raise_remote_failure(
                        "post-load event", event_failures,
                    )
                except RuntimeError as error:
                    event_error = error
            if event_error is None:
                raise RuntimeError("checkpoint post-load event failed")
            execution.poison(event_error, phase="checkpoint post-load event")
            raise event_error

    def _slice(
        self,
        field_name: str,
        info: Any,
        array: np.ndarray,
        indices: np.ndarray,
    ) -> np.ndarray:
        axis = self.model._partition.logical_axis(
            field_name, info, tuple(array.shape),
        )
        slicer = [slice(None)] * array.ndim
        slicer[axis] = indices
        try:
            return array[tuple(slicer)]
        except IndexError as exc:
            raise ValueError(
                f"Cannot shard checkpoint field {field_name!r} with shape "
                f"{array.shape} on logical axis {axis}"
            ) from exc

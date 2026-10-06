# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Construction-input snapshots, equivalent to parameter files.

Persist parameters, topology and initializable physical state, never calendar,
spin-up progress or execution metadata. A new model owns its own run schedule.
One rank-synchronous phase driver saves every world size: each rank writes a
hidden part, and rank zero publishes the single part (one rank) or the merge
of all parts as the checkpoint.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn
from uuid import uuid4

import numpy as np
import torch

from hydroforge.core.errors import (
    ResourceCleanupError,
    distributed_failure_error,
    failure_description,
)
from hydroforge.core.events import emit
from hydroforge.data.input import InputProxy
from hydroforge.io.construction_input import (
    merge_construction_parts,
    write_construction_input,
)
from hydroforge.io.files import atomic_output_path, publish_file


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
    """This rank's host snapshot of one checkpoint, ordered by name."""

    path: Path
    values: dict[str, Any]
    distributed: tuple[str, ...]
    global_fields: tuple[str, ...]
    groups: dict[str, str]


def _torch_numpy_dtype(name: str, dtype: torch.dtype) -> np.dtype:
    """Return the NetCDF-writable NumPy dtype of one Torch dtype."""

    try:
        return torch.empty(0, dtype=dtype, device="cpu").numpy().dtype
    except TypeError as error:
        raise TypeError(
            f"checkpoint construction input {name!r} has dtype {dtype}, "
            "which has no NumPy equivalent"
        ) from error


def _host_copy(name: str, value: Any) -> Any:
    """Detach one construction value into host storage owned by the snapshot."""

    if isinstance(value, torch.Tensor):
        _torch_numpy_dtype(name, value.dtype)
        return value.detach().to(device="cpu", copy=True).numpy()
    if isinstance(value, np.ndarray):
        return np.array(value, order="K", copy=True, subok=False)
    if isinstance(value, np.generic):
        return value.copy()
    if type(value) in {bool, int, float}:
        return value
    raise TypeError(
        f"checkpoint construction input {name!r} has unsupported value type "
        f"{type(value).__name__!r}"
    )


class CheckpointRuntime:
    """Persist complete inputs for constructing a fresh model."""

    def __init__(self, runtime: Any) -> None:
        self.runtime = runtime
        fields = self._compile_input_fields()
        self.plan = _CheckpointPlan(
            fields=fields,
            layout_signature=tuple(
                (
                    field.name,
                    field.module_name,
                    tuple(
                        None
                        if axis == field.partition_axis
                        or (
                            axis == 0
                            and field.info.tensor is not None
                            and len(field.shape) == len(field.info.tensor.shape) + 1
                        )
                        else extent
                        for axis, extent in enumerate(field.shape)
                    ),
                    str(field.numpy_dtype),
                    field.coordinate,
                    field.partition_axis,
                )
                for field in fields
            ),
        )

    def _coordinate_save_entry(
        self,
        error: BaseException | None,
        stage: _CheckpointSaveStage | None,
    ) -> tuple[tuple[dict[str, str] | None, ...], str | None]:
        """Validate the save declaration and publish one rank-zero nonce."""

        plan = self.runtime.plan
        candidate = uuid4().hex if plan.rank == 0 else None
        signature = (
            None
            if stage is None
            else (
                self.plan.layout_signature,
                stage.distributed,
                tuple(sorted(stage.groups.items())),
            )
        )
        failures, payloads = self.runtime.channel.exchange(
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
    def _numpy_dtype(name: str, value: Any) -> np.dtype:
        if isinstance(value, torch.dtype):
            return _torch_numpy_dtype(name, value)
        if isinstance(value, torch.Tensor):
            return _torch_numpy_dtype(name, value.dtype)
        return np.asarray(value).dtype

    def _local_indices(self, name: str) -> np.ndarray | None:
        runtime = self.runtime
        group = runtime.plan.fields.variable_groups.get(name)
        return None if group is None else runtime.partition.rank_indices(group)

    def _local_input_value(self, name: str) -> Any:
        """Reload one rank-local input whose runtime storage was discarded."""

        return self.runtime.input.read_local(name, self._local_indices(name))

    def _field_value(self, field: _InputField) -> Any:
        value = getattr(field.module, field.name)
        if value is None:
            value = self._local_input_value(field.name)
        return value

    def _compile_input_fields(self) -> tuple[_InputField, ...]:
        """Compile every current value needed by a fresh model construction."""

        runtime = self.runtime
        source = runtime.input
        groups = runtime.plan.fields.variable_groups
        fields: dict[str, _InputField] = {}
        for name, spec in source.fields.items():
            info = spec.field
            tensor = info.tensor
            if tensor is not None:
                if tensor.category not in {"param", "topology", "init_state"}:
                    continue
            module = runtime.modules[info.module_name]
            if name in module.nc_excluded_fields or info.excluded:
                continue
            value = getattr(module, name)
            if (
                tensor is None
                and not info.required
                and name not in source
                and type(value) not in {bool, int, float}
            ):
                # Non-numeric declaration defaults are not construction values.
                continue
            metadata = None
            if value is None:
                if name not in source:
                    # A discarded optional tensor that originated from its
                    # declared default is reconstructed from that same default.
                    continue
                # The layout needs only metadata; ``save`` reads the values.
                metadata = source.local_metadata(name, self._local_indices(name))
                if metadata is None:
                    value = self._local_input_value(name)
            if metadata is None:
                shape = tuple(
                    value.shape if isinstance(value, torch.Tensor) else np.shape(value)
                )
                dtype = self._numpy_dtype(name, value)
            else:
                shape = metadata[0]
                dtype = self._numpy_dtype(name, metadata[1])
            coordinate = groups.get(name)
            fields[name] = _InputField(
                name=name,
                module_name=info.module_name,
                module=module,
                info=info,
                shape=shape,
                numpy_dtype=dtype,
                coordinate=coordinate,
                partition_axis=(
                    len(shape) - len(info.tensor.shape)
                    if coordinate is not None
                    else None
                ),
            )
        return tuple(fields[name] for name in sorted(fields))

    def _flush_statistics_output(self) -> None:
        """Make streamed statistics rows durable before a checkpoint commits."""

        statistics = self.runtime.statistics
        if statistics is not None:
            statistics.sink.flush(self.runtime.clock.current_time)

    def _stage_save(self) -> _CheckpointSaveStage:
        """Snapshot complete current construction input without publishing."""

        runtime = self.runtime
        plan = runtime.plan
        clock = runtime.clock
        date = clock.current_time
        timestamp = date.strftime("%Y%m%d_%H%M%S") if date is not None else "latest"
        if plan.schedule is not None:
            timestamp += f"_step{clock.schedule_index}"
        values: dict[str, Any] = {}
        distributed: list[str] = []
        global_fields: list[str] = []
        groups: dict[str, str] = {}

        for field in self.plan.fields:
            if field.coordinate is not None:
                # ``save`` has validated the axis-0 partition layout.
                groups[field.name] = field.coordinate
            elif plan.rank != 0:
                continue
            value = self._field_value(field)
            if field.info.tensor is None and field.name not in runtime.input:
                default = type(field.module).model_fields[field.name].default
                if type(default) in {bool, int, float} and value == default:
                    continue
            values[field.name] = _host_copy(field.name, value)
            (distributed if field.coordinate is not None else global_fields).append(
                field.name
            )

        # Ownership grouping is a construction input even when no module
        # declares it. Preserve its rank-local rows alongside the root axis.
        coordinate = plan.spec.partition_key
        group_name = plan.spec.partition_group
        if coordinate is not None and group_name not in values:
            indices = runtime.partition.rank_indices(coordinate)
            values[group_name] = _host_copy(
                group_name, runtime.input.read_local(group_name, indices)
            )
            groups[group_name] = coordinate
            distributed.append(group_name)

        for coordinate in sorted(set(groups.values())):
            if coordinate in values:
                continue
            entry = runtime.namespace[coordinate]
            values[coordinate] = _host_copy(
                coordinate, getattr(entry.module, entry.field_name)
            )
            groups[coordinate] = coordinate
            distributed.append(coordinate)

        return _CheckpointSaveStage(
            path=plan.output.directory / f"model_state_{timestamp}.nc",
            values=dict(sorted(values.items())),
            distributed=tuple(distributed),
            global_fields=tuple(global_fields),
            groups=groups,
        )

    def _emit_save_events(self, stage: _CheckpointSaveStage) -> None:
        runtime = self.runtime
        for event, message, fields in (
            (
                "checkpoint.saved_distributed",
                "Saved distributed state fields",
                stage.distributed,
            ),
            (
                "checkpoint.saved_global",
                "Saved global state fields",
                stage.global_fields,
            ),
        ):
            if fields:
                emit(
                    runtime,
                    "info",
                    event,
                    message,
                    rank=runtime.plan.rank,
                    fields=fields,
                )

    def _rollback(
        self, primary: BaseException, part: Path, checkpoint_id: str
    ) -> NoReturn:
        """Remove this rank's part on every rank, then raise the primary failure."""

        rollback_error: BaseException | None = None
        try:
            part.unlink(missing_ok=True)
        except BaseException as error:
            rollback_error = error
        rollback_failures = self.runtime.channel.gather(
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
            raise ResourceCleanupError(
                "checkpoint save rollback",
                (primary, rollback_failure),
            ) from primary
        raise primary

    def _phase(
        self,
        phase: str,
        action: Callable[[], Any],
        *,
        label: str,
        signature: tuple[str, ...],
        part: Path | None = None,
    ) -> Any:
        """Run one rank-synchronous save phase and agree on its outcome.

        Until publication a failure on any rank rolls back every rank's
        ``part``; from the commit on it poisons the runtime instead.
        """

        error: BaseException | None = None
        result = None
        try:
            result = action()
        except BaseException as caught:
            error = caught
        failures = self.runtime.channel.gather(
            error, phase=f"checkpoint.save.{phase}", signature=signature
        )
        if not any(failure is not None for failure in failures):
            return result
        failure = (
            error
            if error is not None
            else distributed_failure_error(f"distributed {label}", failures)
        )
        if part is not None:
            self._rollback(failure, part, signature[0])
        self.runtime.execution.poison(failure, phase=label)
        raise failure

    def _collect_parts(self, path: Path, parts: tuple[Path, ...]) -> None:
        """Remove merged parts after the commit; failures only warn."""

        runtime = self.runtime
        errors: list[BaseException] = []
        try:
            emit(
                runtime,
                "info",
                "checkpoint.merged",
                "Merged distributed state",
                rank=0,
                path=path,
            )
        except BaseException as error:
            errors.append(error)
        cleanup_failures = []
        for part in parts:
            try:
                part.unlink(missing_ok=True)
            except BaseException as error:
                cleanup_failures.append(
                    {"path": str(part), **failure_description(error)}
                )
        if cleanup_failures:
            try:
                emit(
                    runtime,
                    "warning",
                    "checkpoint.cleanup_failed",
                    "Merged checkpoint was published but temporary rank "
                    "files could not all be removed",
                    rank=0,
                    failures=tuple(cleanup_failures),
                )
            except BaseException as error:
                errors.append(error)
        if len(errors) == 1:
            raise errors[0]
        if errors:
            raise ResourceCleanupError("checkpoint post-commit events", errors)

    def save(self) -> InputProxy:
        """Persist a checkpoint; return it reopened lazily on every rank."""

        runtime = self.runtime
        plan = runtime.plan
        stage = None
        stage_error: BaseException | None = None
        try:
            runtime.require_healthy(f"{type(runtime.owner).__name__}.save_state")
            # Validate the layout before any side effect (statistics flush).
            for field in self.plan.fields:
                if field.coordinate is not None and field.partition_axis != 0:
                    raise ValueError(
                        f"checkpoint field {field.name!r} is partitioned on axis "
                        f"{field.partition_axis}; construction inputs partition "
                        "axis 0 only"
                    )
            self._flush_statistics_output()
            stage = self._stage_save()
        except BaseException as error:
            stage_error = error
        stage_failures, checkpoint_id = self._coordinate_save_entry(
            stage_error,
            stage,
        )
        if any(failure is not None for failure in stage_failures):
            if stage_error is not None:
                raise stage_error
            raise distributed_failure_error(
                "distributed checkpoint state snapshot",
                stage_failures,
            )
        path = stage.path
        parts = tuple(
            path.with_name(f".{path.name}.{checkpoint_id}.rank{rank}.part")
            for rank in range(plan.world_size)
        )
        part = parts[plan.rank]
        options = plan.output.config.checkpoint_netcdf
        single = plan.world_size == 1
        # Restarts compare the model's parameter options with these.
        fingerprint = {"options": plan.options, "opened_modules": plan.modules}

        def write() -> None:
            if path.exists():
                emit(
                    runtime,
                    "warning",
                    "checkpoint.overwrite",
                    "Overwriting existing model state",
                    rank=plan.rank,
                    path=path,
                )
            # A single part is the checkpoint itself; rank parts are
            # compressed once, by the merge.
            # Rank parts carry no global attributes; the merge records the
            # option fingerprint of the published checkpoint.
            write_construction_input(
                part,
                stage.values,
                partitions=stage.groups,
                netcdf_options=options if single else {},
                **(fingerprint if single else {}),
            )

        def publish() -> None:
            if plan.rank != 0:
                return
            if single:
                publish_file(part, path)
                return
            with atomic_output_path(path) as temporary:
                merge_construction_parts(
                    temporary,
                    parts,
                    partitions=stage.groups,
                    netcdf_options=options,
                    **fingerprint,
                )

        def commit() -> None:
            if not path.is_file():
                raise FileNotFoundError(f"Checkpoint commit point is missing: {path}")

        def collect() -> None:
            if plan.rank == 0 and not single:
                self._collect_parts(path, parts)

        before = (checkpoint_id,)
        after = (checkpoint_id, str(path))
        self._phase(
            "write",
            write,
            label="checkpoint rank write",
            signature=before,
            part=part,
        )
        self._phase(
            "events.precommit",
            lambda: self._emit_save_events(stage),
            label="checkpoint pre-commit event",
            signature=before,
            part=part,
        )
        self._phase(
            "publish",
            publish,
            label="checkpoint publication",
            signature=after,
            part=part,
        )
        self._phase(
            "commit",
            commit,
            label="checkpoint save commit",
            signature=after,
        )
        # The published file is the commit point.  Removing merged parts is
        # garbage collection that only warns: a published checkpoint must not
        # be reported as a failed save.
        self._phase(
            "events.postcommit",
            collect,
            label="checkpoint post-commit event",
            signature=before,
        )
        return self._phase(
            "reopen",
            lambda: InputProxy.from_nc(path, lazy=True),
            label="checkpoint save reopen",
            signature=after,
        )

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime, timedelta
from enum import Enum
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast
from uuid import uuid4

import cftime
import numpy as np
import torch
import torch.distributed as dist

from hydroforge.contracts.errors import (
    ResourceCleanupError,
    distributed_failure_error,
    failure_description,
)
from hydroforge.contracts.events import emit
from hydroforge.contracts.temporal import SimulationSchedule, canonical_calendar
from hydroforge.contracts.validation import HydroForgeModel

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel


def _distributed_array_signature(value: np.ndarray) -> tuple[Any, ...]:
    array = np.asarray(value)
    canonical = np.ascontiguousarray(array)
    return (
        "numpy",
        canonical.dtype.str,
        tuple(array.shape),
        sha256(canonical.view(np.uint8).tobytes()).hexdigest(),
    )


def _distributed_date_signature(
    value: datetime | cftime.datetime | None,
) -> tuple[Any, ...] | None:
    if value is None:
        return None
    return (
        _qualified_type_name(type(value)),
        canonical_calendar(getattr(value, "calendar", "standard")),
        value.year,
        value.month,
        value.day,
        value.hour,
        value.minute,
        value.second,
        value.microsecond,
        getattr(value, "fold", None),
        getattr(value, "has_year_zero", None),
    )


def _distributed_schedule_signature(
    schedule: SimulationSchedule | None,
) -> tuple[Any, ...] | None:
    """Digest explicit schedules without publishing every step object."""
    if schedule is None:
        return None
    if schedule._is_regular:
        return (
            "regular",
            schedule.calendar,
            _distributed_date_signature(schedule.regular_start),
            _distributed_date_signature(schedule.regular_end),
            _distributed_value_signature(schedule.regular_step),
            _distributed_value_signature(schedule.source_interval),
            _distributed_value_signature(schedule.spinup),
            schedule._num_spinup_steps,
            schedule.num_main_steps,
        )
    digest = sha256()
    for step in schedule.explicit_steps:
        encoded = json.dumps(
            _distributed_value_signature(step),
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return (
        "explicit",
        schedule.calendar,
        len(schedule.explicit_steps),
        _distributed_date_signature(schedule.execution_start),
        _distributed_date_signature(schedule._end),
        digest.hexdigest(),
    )


def _distributed_tensor_signature(value: torch.Tensor) -> tuple[Any, ...]:
    canonical = value.detach().to(device="cpu").contiguous().reshape(-1)
    payload = canonical.view(torch.uint8).numpy().tobytes()
    return ("torch", str(value.dtype), tuple(value.shape), sha256(payload).hexdigest())


def _distributed_value_signature(value: Any) -> Any:
    """Encode one declaration as stable, equality-safe Python primitives."""
    if value is None or type(value) in {bool, int, str}:
        return value
    if type(value) is float:
        return ("float", value.hex())
    if isinstance(value, Enum):
        return (
            "enum",
            _qualified_type_name(type(value)),
            _distributed_value_signature(value.value),
        )
    if isinstance(value, (datetime, cftime.datetime)):
        return _distributed_date_signature(value)
    if type(value) is timedelta:
        return ("timedelta", value.days, value.seconds, value.microseconds)
    if isinstance(value, Path):
        return ("path", str(value.absolute()))
    if isinstance(value, torch.device):
        return ("device", value.type)
    if isinstance(value, torch.dtype):
        return ("dtype", str(value))
    if isinstance(value, torch.Tensor):
        return _distributed_tensor_signature(value)
    if isinstance(value, np.ndarray):
        return _distributed_array_signature(value)
    if isinstance(value, np.generic):
        return _distributed_array_signature(np.asarray(value))
    if isinstance(value, HydroForgeModel):
        fields = object.__getattribute__(value, "__dict__")
        return (
            "model",
            _qualified_type_name(type(value)),
            tuple(
                (name, _distributed_value_signature(fields[name]))
                for name in type(value).model_fields
            ),
        )
    if isinstance(value, Mapping):
        entries = tuple(
            (
                (_distributed_value_signature(key), _distributed_value_signature(item))
                for key, item in value.items()
            )
        )
        return ("mapping", tuple(sorted(entries, key=lambda item: repr(item[0]))))
    if isinstance(value, tuple):
        return ("tuple", tuple(map(_distributed_value_signature, value)))
    if isinstance(value, (set, frozenset)):
        items = tuple(map(_distributed_value_signature, value))
        return ("set", tuple(sorted(items, key=repr)))
    if isinstance(value, bytes):
        return ("bytes", value.hex())
    raise TypeError(
        f"distributed runtime declarations cannot contain unsupported value {type(value).__name__}"
    )


def _qualified_type_name(value: type[Any]) -> str:
    return f"{value.__module__}.{value.__qualname__}"


class RuntimeLifecycle:
    """Own materialization, cross-rank health coordination and ordered teardown."""

    def __init__(self, model: AbstractModel) -> None:
        self.model = model

    def distributed_input_schema_signature(self) -> tuple[Any, ...]:
        """Describe external storage shape without hashing physical fields."""
        model = self.model
        proxy = model.input_proxy
        return tuple(
            (
                name,
                model._input.get_var_shape(name),
                str(proxy._get_var_dtype(name)),
                model._semantic_plan.input_axes.get(name),
                model._semantic_plan.variable_groups.get(name),
            )
            for name in sorted(model._input.fields)
            if name in model._input
        )

    def distributed_input_storage_signature(self) -> tuple[Any, ...]:
        """Identify active resident values and lazy source declarations."""
        model = self.model
        proxy = model.input_proxy
        resident = dict(proxy._resident_items())
        fields: list[tuple[Any, ...]] = []
        for name in sorted(model._input.fields):
            if name not in model._input:
                continue
            if name in resident:
                fields.append(
                    (name, "resident", _distributed_value_signature(resident[name]))
                )
                continue
            source = proxy.sources[name]
            identity = source.file_identity
            fields.append(
                (
                    name,
                    "netcdf",
                    source.dimensions,
                    source.shape,
                    source.dtype,
                    source.alignment_dim,
                    _distributed_value_signature(source.alignment_indices),
                    identity.size,
                    identity.mtime_ns,
                )
            )
        return (tuple(fields), tuple(sorted(proxy.injected_vars)))

    def distributed_partition_identity_signature(self) -> tuple[Any, ...]:
        """Hash values that decide rank ownership and reference routing."""
        model = self.model
        schema = model._semantic_plan.partition_schema
        names = set(schema.coordinates)
        if model.partition_group in model._input:
            names.add(model.partition_group)
        for name, metadata in schema.fields.items():
            if metadata.references or metadata.partition_by or metadata.selects:
                names.add(name)
        proxy = model.input_proxy
        return tuple(
            (name, _distributed_value_signature(proxy._get_value_trusted(name)))
            for name in sorted(names)
            if name in proxy
        )

    def distributed_runtime_declaration_signature(self) -> tuple[Any, ...]:
        """Return the complete rank-shared model control-plane identity."""
        model = self.model
        module_types = model._module_types()
        return (
            ("model", _qualified_type_name(type(model))),
            (
                "modules",
                tuple(
                    (name, _qualified_type_name(module_types[name]))
                    for name in model.opened_modules
                ),
            ),
            ("module_order", model._module_order),
            ("partition", model.partition_key, model.partition_group),
            ("cuda_catalogs", model.cuda_extension_modules),
            ("backend", model._backend),
            ("device_type", model.device.type),
            ("precision", model.precision, model.mixed_precision),
            ("options", _distributed_value_signature(model.options)),
            ("execution", model.execution_mode, model.BLOCK_SIZE),
            (
                "parallel",
                None if model.parallel is None else model.parallel.model_dump(),
            ),
            (
                "step_field_expressions",
                tuple(
                    (name, expression.source)
                    for name, expression in sorted(
                        model._step_field_expressions.items()
                    )
                ),
            ),
            (
                "step_field_providers",
                tuple(
                    (
                        name,
                        getattr(provider, "__module__", type(provider).__module__),
                        getattr(provider, "__qualname__", type(provider).__qualname__),
                    )
                    for name, provider in sorted(model._step_field_providers.items())
                ),
            ),
            (
                "ensemble",
                model.ensemble_size,
                _distributed_value_signature(model.ensemble_forcing_fields),
            ),
            (
                "output",
                model.experiment_name,
                str((model.output_dir / model.experiment_name).absolute()),
                model.output_workers,
                model.output_split_by_year,
                model.max_pending_steps,
                model.max_pending_output_bytes,
                model.save_kernels,
                model.in_memory_output,
                cast(torch.device, model.result_device).type,
            ),
            ("calendar", model.calendar),
            ("initial_time", _distributed_date_signature(model.initial_time)),
            ("schedule", _distributed_schedule_signature(model.simulation_schedule)),
            (
                "statistics",
                _distributed_value_signature(model.variables_to_save),
                _distributed_value_signature(model._statistics_plan),
                model.statistics_save_precision,
            ),
            (
                "netcdf",
                _distributed_value_signature(model.output_netcdf_options),
                _distributed_value_signature(model.checkpoint_netcdf_options),
            ),
            (
                "parameter_changes",
                _distributed_value_signature(model.parameter_changes),
            ),
            ("input_schema", model._distributed_input_schema_signature()),
            ("input_storage", model._distributed_input_storage_signature()),
            ("partition_identity", model._distributed_partition_identity_signature()),
        )

    def distributed_compiled_runtime_signature(self) -> tuple[Any, ...]:
        """Describe rank-invariant services produced by initialization."""
        model = self.model
        return (
            model._execution.backend,
            model._execution.capture_mode,
            model._checkpoint.plan.layout_signature,
            tuple(
                sorted(
                    descriptor.protocol_name
                    for descriptor in model._execution.step_policies
                )
            ),
        )

    def materialize_runtime(self) -> None:
        """Build private runtime services from a validated model identity."""
        model = self.model
        from hydroforge.compiler.initialization import ModelInitializer

        schedule = model.simulation_schedule
        model._current_time = (
            schedule.execution_start if schedule is not None else model.initial_time
        )
        model._prepare_output_directory()
        ModelInitializer(model).run()

    def ensure_runtime_materialized(self) -> None:
        """Materialize runtime state as one rank-synchronous transaction."""
        model = self.model
        if model._runtime_materialized:
            return
        if model.world_size > 1:
            model._coordinate_runtime_materialization_preflight()
        initialization_error: BaseException | None = None
        try:
            model._materialize_runtime()
        except BaseException as error:
            initialization_error = error
        if model.world_size > 1:
            model._coordinate_runtime_materialization(initialization_error)
            return
        if initialization_error is not None:
            model._discard_runtime_materialization()
            raise initialization_error
        model._runtime_materialized = True

    def coordinate_runtime_materialization_preflight(self) -> None:
        """Prove every rank will initialize the same runtime transaction."""
        model = self.model
        signature: tuple[Any, ...] | None = None
        local_error: BaseException | None = None
        try:
            signature = model._distributed_runtime_declaration_signature()
        except BaseException as error:
            local_error = error
        failures = model._gather_distributed_failures(
            local_error, phase="runtime.materialization.preflight", signature=signature
        )
        if not any(failure is not None for failure in failures):
            return
        if local_error is not None:
            raise local_error
        raise distributed_failure_error(
            "distributed model runtime declaration validation", failures
        )

    def install_distributed_output_run_id(self, payloads: tuple[Any, ...]) -> None:
        """Install the rank-zero output identity exchanged at runtime commit."""
        model = self.model
        if len(payloads) != model.world_size:
            raise RuntimeError(
                "distributed runtime materialization returned an invalid output run-ID payload count"
            )
        run_id = payloads[0]
        if not isinstance(run_id, str) or not run_id:
            raise RuntimeError(
                "distributed runtime materialization did not publish a non-empty output run ID from rank zero"
            )
        if any(payload is not None for payload in payloads[1:]):
            raise RuntimeError(
                "distributed runtime materialization received output run IDs from nonzero ranks"
            )
        statistics = getattr(model, "_statistics", None)
        aggregator = getattr(statistics, "aggregator", None)
        if aggregator is not None:
            aggregator.run_id = run_id

    def gather_distributed_failures(
        self,
        error: BaseException | None,
        *,
        phase: str,
        signature: tuple[Any, ...] | None = None,
    ) -> tuple[dict[str, str] | None, ...]:
        """Publish one phase-tagged public transaction result to every rank."""
        model = self.model
        failures, _payloads = model._exchange_distributed_public_transaction(
            error, phase=phase, signature=signature
        )
        return failures

    def exchange_distributed_public_transaction(
        self,
        error: BaseException | None,
        *,
        phase: str,
        signature: tuple[Any, ...] | None = None,
        payload: Any = None,
    ) -> tuple[tuple[dict[str, str] | None, ...], tuple[Any, ...]]:
        """Exchange one tagged transaction record and optional phase payload."""
        model = self.model
        if not dist.is_available() or not dist.is_initialized():
            raise RuntimeError(
                "multi-rank model transactions require an initialized torch.distributed process group"
            )
        if type(phase) is not str or not phase:
            raise RuntimeError(
                "distributed public transaction phase must be a non-empty string"
            )
        sequence = model._distributed_public_sequence
        model._distributed_public_sequence = sequence + 1
        local = (
            sequence,
            phase,
            signature,
            None if error is None else failure_description(error),
            payload,
        )
        observed: list[Any] = [None] * model.world_size
        dist.all_gather_object(observed, local)
        if any(
            not isinstance(value, tuple)
            or len(value) != 5
            or type(value[0]) is not int
            or (type(value[1]) is not str)
            for value in observed
        ):
            raise RuntimeError(
                f"distributed public transaction protocol received a malformed record: {observed!r}"
            )
        identities = tuple((value[0], value[1]) for value in observed)
        if len(set(identities)) != 1:
            raise RuntimeError(
                f"distributed public phase mismatch across ranks: {identities!r}"
            )
        failures = tuple(value[3] for value in observed)
        if not any(failure is not None for failure in failures):
            signatures = tuple(value[2] for value in observed)
            if any(value != signatures[0] for value in signatures[1:]):
                raise RuntimeError(
                    f"distributed public transaction inputs differ across ranks during {phase!r}: {signatures!r}"
                )
        return (
            cast(tuple[dict[str, str] | None, ...], failures),
            tuple(value[4] for value in observed),
        )

    def release_runtime_materialization(self) -> None:
        """Release every service created by one materialization attempt."""
        model = self.model
        failures: list[BaseException] = []
        statistics = getattr(model, "_statistics", None)
        execution = getattr(model, "_execution", None)
        for resource in (statistics, execution):
            if resource is not None:
                try:
                    resource.close()
                except BaseException as error:
                    failures.append(error)
        model._discard_runtime_materialization()
        model._runtime_materialized = False
        if failures:
            error = ResourceCleanupError("model resources", failures)
            raise error from failures[0]

    def coordinate_runtime_materialization(
        self, initialization_error: BaseException | None
    ) -> None:
        """Commit initialization only after every rank reports success."""
        model = self.model
        compiled_signature: tuple[Any, ...] | None = None
        if initialization_error is None:
            try:
                compiled_signature = model._distributed_compiled_runtime_signature()
            except BaseException as error:
                initialization_error = error
        run_id_candidate: str | None = None
        if initialization_error is None and model.rank == 0:
            try:
                run_id_candidate = str(uuid4())
            except BaseException as error:
                initialization_error = error
        try:
            initialization_failures, run_id_payloads = (
                model._exchange_distributed_public_transaction(
                    initialization_error,
                    phase="runtime.materialization",
                    signature=compiled_signature,
                    payload=run_id_candidate,
                )
            )
        except BaseException as coordination_error:
            cleanup_error: BaseException | None = None
            try:
                model._release_runtime_materialization()
            except BaseException as error:
                cleanup_error = error
            failures = tuple(
                error
                for error in (initialization_error, coordination_error, cleanup_error)
                if error is not None
            )
            if len(failures) == 1:
                raise failures[0]
            error = ResourceCleanupError(
                "distributed model initialization coordination", failures
            )
            raise error from coordination_error
        if not any(failure is not None for failure in initialization_failures):
            try:
                model._install_distributed_output_run_id(run_id_payloads)
            except BaseException as error:
                initialization_error = error
                description = failure_description(error)
                initialization_failures = tuple(
                    description for _ in range(model.world_size)
                )
            else:
                model._runtime_materialized = True
                return
        cleanup_error: BaseException | None = None
        try:
            model._release_runtime_materialization()
        except BaseException as error:
            cleanup_error = error
        primary = (
            initialization_error
            if initialization_error is not None
            else distributed_failure_error(
                "distributed model initialization", initialization_failures
            )
        )
        try:
            cleanup_failures = model._gather_distributed_failures(
                cleanup_error, phase="runtime.materialization.cleanup"
            )
        except BaseException as coordination_error:
            failures = [primary, coordination_error]
            if cleanup_error is not None:
                failures.append(cleanup_error)
            error = ResourceCleanupError(
                "distributed model initialization cleanup coordination", failures
            )
            raise error from primary
        if any(failure is not None for failure in cleanup_failures):
            cleanup_failure = (
                cleanup_error
                if cleanup_error is not None
                else distributed_failure_error(
                    "distributed model initialization cleanup", cleanup_failures
                )
            )
            error = ResourceCleanupError(
                "distributed model initialization rollback", (primary, cleanup_failure)
            )
            raise error from primary
        raise primary

    def discard_runtime_materialization(self) -> None:
        """Remove every object derived from one materialized runtime."""
        model = self.model
        model._data.release()
        model._modules.clear()
        model._module_links = None
        for name in ("_variable_map", "group_id_to_rank"):
            model.__dict__.pop(name, None)

    def ensure_healthy_runtime(self) -> None:
        """Enter trusted runtime only after a public request has validated."""
        model = self.model
        if model.parallel is not None:
            model.parallel.validate_live()
        model._ensure_runtime_materialized()
        failure = model._execution.failure
        if failure is not None:
            raise model._execution.poisoned_error(failure)

    def prepare_output_directory(self) -> None:
        """Acquire the output directory only after validation completes."""
        model = self.model
        if model.spatial_rank != 0:
            return
        if not model.output_full_dir.exists():
            model.output_full_dir.mkdir(parents=True, exist_ok=True)
            return
        emit(
            model,
            "warning",
            "output.directory_exists",
            "Output directory already exists; contents may be overwritten",
            directory=model.output_full_dir,
        )

    def close(self) -> None:
        """Atomically release output workers and backend execution resources."""
        model = self.model
        if not model._runtime_materialized:
            return
        model._release_runtime_materialization()

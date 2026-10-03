"""Rank-shared identities of a model declaration and its materialization."""

from __future__ import annotations

import json
from hashlib import sha256
from typing import TYPE_CHECKING, Any

from hydroforge.contracts.schedule import SimulationSchedule
from hydroforge.core.identity import canonical, tensor_content

if TYPE_CHECKING:
    from hydroforge.execution.session import ModelRuntime


def signature_value(value: Any) -> Any:
    """Encode one declaration value; tensors are identified by content."""

    return canonical(value, tensor=tensor_content)


def _schedule_signature(schedule: SimulationSchedule | None) -> Any:
    """Digest explicit schedules without publishing every step object."""
    if schedule is None:
        return None
    if schedule._is_regular:
        return (
            "regular",
            schedule.calendar,
            signature_value(schedule.regular_start),
            signature_value(schedule.regular_end),
            signature_value(schedule.regular_step),
            signature_value(schedule.source_interval),
            signature_value(schedule.spinup),
            schedule._num_spinup_steps,
            schedule.num_main_steps,
        )
    digest = sha256()
    for step in schedule.explicit_steps:
        encoded = json.dumps(
            signature_value(step), ensure_ascii=False, separators=(",", ":")
        ).encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return (
        "explicit",
        schedule.calendar,
        len(schedule.explicit_steps),
        signature_value(schedule._start),
        signature_value(schedule._end),
        digest.hexdigest(),
    )


def _parameter_changes_signature(runtime: ModelRuntime) -> Any:
    """Encode the declared changes with their schedule-calendar start dates."""

    starts = {id(target.change): target.start for target in runtime.plan.parameters}
    return tuple(
        (signature_value(change), signature_value(starts[id(change)]))
        for change in runtime.owner.parameter_changes
    )


def _input_schema_signature(runtime: ModelRuntime) -> tuple[Any, ...]:
    """Describe external storage shape without hashing physical fields."""
    source = runtime.input
    groups = runtime.plan.fields.variable_groups
    return tuple(
        (
            name,
            source.get_var_shape(name),
            str(source.proxy._dtype(name)),
            source.axes.get(name),
            groups.get(name),
        )
        for name in sorted(source.fields)
        if name in source
    )


def _input_storage_signature(runtime: ModelRuntime) -> tuple[Any, ...]:
    """Identify active resident values and lazy source declarations."""
    source = runtime.input
    proxy = source.proxy
    resident = dict(proxy._resident_items())
    fields: list[tuple[Any, ...]] = []
    for name in sorted(source.fields):
        if name not in source:
            continue
        if name in resident:
            fields.append((name, "resident", signature_value(resident[name])))
            continue
        declared = proxy.sources[name]
        identity = declared.file_identity
        fields.append(
            (
                name,
                "netcdf",
                declared.dimensions,
                declared.shape,
                declared.dtype,
                declared.alignment_dim,
                signature_value(declared.alignment_indices),
                identity.size,
                identity.mtime_ns,
            )
        )
    return (tuple(fields), tuple(sorted(proxy.injected_vars)))


def _partition_identity_signature(runtime: ModelRuntime) -> tuple[Any, ...]:
    """Hash values that decide rank ownership and reference routing."""
    plan = runtime.plan
    schema = plan.fields.partition
    names = set(schema.coordinates)
    if plan.spec.partition_group in runtime.input:
        names.add(plan.spec.partition_group)
    for name, metadata in schema.fields.items():
        if metadata.references or metadata.partition_by or metadata.selects:
            names.add(name)
    source = runtime.input
    return tuple(
        (name, signature_value(source.value(name)))
        for name in sorted(names)
        if name in source
    )


def declaration_signature(runtime: ModelRuntime) -> tuple[Any, ...]:
    """Return the complete rank-shared model control-plane identity."""
    plan = runtime.plan
    model = runtime.owner
    modules = plan.spec.modules
    step_fields = plan.step_fields
    output = plan.output
    config = output.config
    return (
        ("model", plan.model),
        (
            "modules",
            tuple(
                (
                    name,
                    modules[name].module_type.__module__,
                    modules[name].module_type.__qualname__,
                )
                for name in plan.modules
            ),
        ),
        ("module_order", plan.module_order),
        ("partition", plan.spec.partition_key, plan.spec.partition_group),
        ("backend", plan.backend.name),
        ("device_type", plan.device.type),
        ("precision", plan.precision, plan.mixed_precision, plan.metal_emulation),
        ("options", signature_value(plan.options)),
        ("execution", model.execution_mode, plan.block_size),
        (
            "parallel",
            None if plan.parallel is None else plan.parallel.model_dump(),
        ),
        (
            "step_field_expressions",
            tuple(
                (name, expression.source)
                for name, expression in sorted(step_fields.expressions.items())
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
                for name, provider in sorted(step_fields.providers.items())
            ),
        ),
        (
            "ensemble",
            plan.ensemble_size,
            signature_value(model.ensemble_forcing_fields),
        ),
        (
            "output",
            config.experiment,
            None if config.dir is None else str(config.dir.absolute()),
            config.workers,
            config.split_by_year,
            config.max_pending_steps,
            config.sink,
            config.result_device.type,
        ),
        ("calendar", plan.calendar),
        ("initial_time", signature_value(plan.initial_time)),
        ("schedule", _schedule_signature(plan.schedule)),
        (
            "statistics",
            signature_value(config.variables),
            signature_value(output.windows),
            config.save_precision,
        ),
        (
            "netcdf",
            signature_value(config.netcdf),
            signature_value(config.checkpoint_netcdf),
        ),
        ("parameter_changes", _parameter_changes_signature(runtime)),
        ("input_schema", _input_schema_signature(runtime)),
        ("input_storage", _input_storage_signature(runtime)),
        ("partition_identity", _partition_identity_signature(runtime)),
    )


def compiled_signature(runtime: ModelRuntime) -> tuple[Any, ...]:
    """Describe rank-invariant services produced by materialization."""
    execution = runtime.execution
    return (
        execution.backend.name,
        execution.executor.name,
        runtime.checkpoint.plan.layout_signature,
        tuple(
            sorted(descriptor.protocol_name for descriptor in execution.step_policies)
        ),
    )

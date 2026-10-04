"""Parameter stage: resolve scheduled changes to fields and main steps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import cftime
import torch

from hydroforge.compiler.fields import FieldEntry, FieldPlan
from hydroforge.compiler.partition import coordinate_identity
from hydroforge.compiler.selection import Selection
from hydroforge.contracts.fields import concrete_tensor_dtype
from hydroforge.contracts.parameters import ParameterChange, validate_parameter_scalar
from hydroforge.core.time import DateLike, normalize_calendar_dates

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


@dataclass(frozen=True, slots=True)
class ParameterTarget:
    """One scheduled change bound to its field and schedule position.

    ``target_ids`` and ``id_field`` are the requested global IDs and their
    validated key field; rank-local indices and values are resolved when the
    runtime materializes its inputs.
    """

    change: ParameterChange
    field: FieldEntry
    start: DateLike
    target_ids: tuple[int, ...] | None
    id_field: FieldEntry | None

    @property
    def is_set_value(self) -> bool:
        return self.change._trusted_value("target_value") is not None


def _resolve(
    names: Mapping[str, FieldEntry],
    name: str,
    *,
    owner_module: str | None = None,
    label: str,
) -> FieldEntry:
    entry = None
    if "." not in name and owner_module is not None:
        entry = names.get(f"{owner_module}.{name}")
    if entry is None:
        entry = names.get(name)
    if entry is None:
        raise ValueError(
            f"{label} {name!r} was not found unambiguously in an opened module"
        )
    return entry


def _id_field(
    names: Mapping[str, FieldEntry], change: ParameterChange, field: FieldEntry
) -> FieldEntry:
    tensor = field.tensor
    if not tensor.shape:
        raise ValueError(
            f"parameter change variable {change.variable!r} is scalar and "
            "cannot use target_ids"
        )
    id_name = change.target_id_field or tensor.dim_coords
    if id_name is None:
        raise ValueError(
            f"parameter change variable {change.variable!r} needs "
            "target_id_field because it has no dim_coords"
        )
    id_field = _resolve(
        names, id_name, owner_module=field.module, label="parameter target ID field"
    )
    id_tensor = id_field.tensor
    if not id_tensor.is_key:
        raise ValueError(
            f"parameter target ID field {id_name!r} must declare is_key=True"
        )
    if len(id_tensor.shape) != 1:
        raise ValueError(
            f"parameter target ID field {id_name!r} must be one-dimensional"
        )
    coordinate = (
        None
        if tensor.dim_coords is None
        else coordinate_identity(tensor.dim_coords, names.values())
    )
    id_coordinate = (
        id_field.qualified
        if id_tensor.is_coordinate
        else coordinate_identity(id_tensor.dim_coords, names.values())
        if id_tensor.dim_coords
        else None
    )
    if coordinate != id_coordinate:
        raise ValueError(
            f"parameter target ID field {id_name!r} is not aligned to "
            f"{change.variable!r} coordinate {coordinate!r}"
        )
    return id_field


def _target(
    selection: Selection, fields: FieldPlan, change: ParameterChange
) -> ParameterTarget:
    schedule = selection.schedule
    label = f"parameter change {change.variable!r} start"
    _calendar, normalized, _defaulted = normalize_calendar_dates(
        {label: change.start, "schedule start": schedule._start},
        calendar=schedule.calendar,
        preserve_cftime_declaration=isinstance(schedule._start, cftime.datetime),
    )
    start = normalized[label]
    try:
        main_index = schedule._main_index_at(start)
    except KeyError:
        raise ValueError(
            f"parameter change {change.variable!r} start {start!r} is not a "
            "main simulation step boundary"
        ) from None
    remaining_steps = schedule.num_main_steps - main_index
    if change.active_steps > remaining_steps:
        raise ValueError(
            f"parameter change {change.variable!r} active_steps="
            f"{change.active_steps} exceeds the {remaining_steps} "
            "main simulation step(s) remaining from its start"
        )
    field = _resolve(fields.names, change.variable, label="parameter change variable")
    tensor = field.tensor
    if tensor.category != "param":
        raise ValueError(
            f"parameter change variable {change.variable!r} declares "
            f"category={tensor.category!r}, expected 'param'"
        )
    if tensor.mode == "discard":
        raise ValueError(
            f"parameter change variable {change.variable!r} cannot use mode='discard'"
        )
    value = change._trusted_value("target_value")
    is_set = value is not None
    if not is_set:
        value = change._trusted_value("delta")
    if not isinstance(value, torch.Tensor):
        validate_parameter_scalar(
            value,
            dtype=concrete_tensor_dtype(
                tensor.dtype, selection.dtype, selection.mixed_precision
            ),
            variable_name=change.variable,
            is_set=is_set,
        )
    requested = change._trusted_value("target_ids")
    return ParameterTarget(
        change=change,
        field=field,
        start=start,
        target_ids=(
            None
            if requested is None
            else requested
            if isinstance(requested, tuple)
            else tuple(requested.tolist())
        ),
        id_field=None if requested is None else _id_field(fields.names, change, field),
    )


def validate_set_targets(
    targets: Iterable[tuple[ParameterTarget, str | None, Iterable[int] | None]],
) -> None:
    """Reject SET overlap within each field, start and index namespace.

    Declaration IDs are comparable only within the same key field. After
    materialization (and structural rebinding), all resolved row indices use
    the same namespace. Global SETs conflict across every namespace.
    """

    groups: dict[tuple[str, DateLike], dict[str | None, set[int]] | None] = {}
    for target, namespace, indices in targets:
        if not target.is_set_value:
            continue
        key = (target.field.qualified, target.start)
        if key in groups and (groups[key] is None or indices is None):
            raise ValueError(
                f"parameter {target.change.variable!r} has overlapping SET "
                f"plans at {target.start}: a global SET conflicts "
                "with every other SET"
            )
        if indices is None:
            groups[key] = None
            continue
        namespaces = groups.setdefault(key, {})
        used = namespaces.setdefault(namespace, set())
        selected = set(indices)
        if not used.isdisjoint(selected):
            raise ValueError(
                f"parameter {target.change.variable!r} has overlapping SET "
                f"targets at {target.start}"
            )
        used.update(selected)


def plan_parameters(
    selection: Selection,
    fields: FieldPlan,
    changes: tuple[ParameterChange, ...],
) -> tuple[ParameterTarget, ...]:
    """Resolve declarations and reject SET conflicts known without input I/O."""

    if not changes:
        return ()
    if selection.schedule is None:
        raise ValueError(
            "parameter_changes require simulation_schedule so every "
            "change can be resolved to an exact managed step"
        )
    targets = tuple(
        sorted(
            (_target(selection, fields, change) for change in changes),
            key=lambda target: target.start,
        )
    )
    validate_set_targets(
        (
            target,
            None if target.id_field is None else target.id_field.qualified,
            target.target_ids,
        )
        for target in targets
    )
    return targets

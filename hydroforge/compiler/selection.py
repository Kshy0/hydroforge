"""First compilation stage: modules, backend, precision, clock and topology."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from graphlib import CycleError, TopologicalSorter
from typing import TYPE_CHECKING, Literal

import torch

from hydroforge.contracts.conditions import resolve_conditions
from hydroforge.contracts.fields import tensor_is_active
from hydroforge.contracts.options import OptionsConfig
from hydroforge.contracts.runtime import (
    DEFAULT_BACKEND_REQUIREMENT,
    DEFAULT_MODULE_REQUIREMENT,
    BackendRequirement,
)
from hydroforge.contracts.schedule import SimulationSchedule
from hydroforge.core.time import DateLike, normalize_calendar_dates
from hydroforge.declare.spec import ModelSpec
from hydroforge.parallel.distributed import ProcessTopology
from hydroforge.parallel.mesh import EnsembleParallel
from hydroforge.platform.backend import Backend, resolve_backend
from hydroforge.platform.devices import float64_supported

if TYPE_CHECKING:
    from hydroforge.declare.model import ModelDeclaration


@dataclass(frozen=True, slots=True)
class Selection:
    """Resolved model-wide choices of one declaration."""

    model: str
    spec: ModelSpec
    modules: tuple[str, ...]
    module_order: tuple[str, ...]
    options: OptionsConfig
    conditions: Mapping[str, bool]
    backend: Backend
    device: torch.device
    precision: Literal["float32", "float64"]
    dtype: torch.dtype
    mixed_precision: bool
    metal_emulation: Literal["native", "float32x2"]
    backend_requirement: BackendRequirement
    block_size: int | None
    capture: bool
    calendar: str
    initial_time: DateLike | None
    schedule: SimulationSchedule | None
    rank: int
    world_size: int
    spatial_rank: int
    spatial_world_size: int
    ensemble_size: int | None
    local_ensemble_size: int | None
    member_ids: tuple[int, ...] | None
    parallel: EnsembleParallel | None


def resolve_block_size(
    backend: Backend,
    requirement: BackendRequirement,
    configured: int | None,
    *,
    kernel: int | None = None,
) -> int:
    """Resolve the launch width one backend uses under a model requirement.

    The backend's fixed width wins; otherwise a configured model width
    overrides ``kernel`` (a per-kernel width), which overrides the backend
    default.
    """

    name = backend.name
    value = backend.block.resolve(configured, kernel=kernel, backend=name)
    if requirement.min_block_size is not None and value < requirement.min_block_size:
        raise ValueError(
            f"backend {name!r} requires BLOCK_SIZE >= "
            f"{requirement.min_block_size}, got {value}"
        )
    if requirement.max_block_size is not None and value > requirement.max_block_size:
        raise ValueError(
            f"backend {name!r} requires BLOCK_SIZE <= "
            f"{requirement.max_block_size}, got {value}"
        )
    if requirement.block_size is not None and value != requirement.block_size:
        raise ValueError(
            f"backend {name!r} requires BLOCK_SIZE={requirement.block_size}, "
            f"got {value}"
        )
    return value


def _validate_ensemble_forcing(
    spec: ModelSpec, declaration: ModelDeclaration, conditions: Mapping[str, bool]
) -> None:
    fields = declaration.ensemble_forcing_fields
    if fields and declaration.ensemble_size is None:
        raise ValueError("ensemble_forcing_fields require ensemble_size")
    opened = declaration.opened_modules
    for module_name, field_names in fields.items():
        if module_name not in opened:
            raise ValueError(f"member forcing module {module_name!r} is not open")
        for field_name in field_names:
            field = spec.modules[module_name].tensor_fields.get(field_name)
            if field is None:
                raise ValueError(
                    f"unknown member forcing field {module_name}.{field_name}"
                )
            if field.tensor.category != "forcing":
                raise ValueError(
                    f"member forcing field {module_name}.{field_name} has category {field.tensor.category!r}, expected 'forcing'"
                )
            if not tensor_is_active(field.tensor, opened, conditions=conditions):
                raise ValueError(
                    f"member forcing field {module_name}.{field_name} is inactive"
                )


def _module_order(spec: ModelSpec, opened: tuple[str, ...]) -> tuple[str, ...]:
    """Return the deterministic dependency order of the opened modules."""

    sorter: TopologicalSorter[str] = TopologicalSorter()
    for name in opened:
        sorter.add(
            name,
            *(
                reference
                for reference in spec.modules[name].references
                if reference in opened
            ),
        )
    try:
        return tuple(sorter.static_order())
    except CycleError as error:
        raise ValueError(
            f"opened module references must form an acyclic construction graph: {error.args[1]}"
        ) from error


def select(spec: ModelSpec, declaration: ModelDeclaration) -> Selection:
    """Resolve every model-wide choice and validate their combination."""

    parallel = declaration.parallel
    ensemble_size = declaration.ensemble_size
    if parallel is not None:
        parallel.validate_live()
        if ensemble_size != parallel.ensemble_size:
            raise ValueError("model ensemble_size must match the ensemble process mesh")
    opened = declaration.opened_modules
    conditions = resolve_conditions(
        (spec.modules[name] for name in opened), declaration.options
    )
    _validate_ensemble_forcing(spec, declaration, conditions)
    schedule = declaration.simulation_schedule
    for name in opened:
        rule = spec.module_requirements.get(name, DEFAULT_MODULE_REQUIREMENT)
        if not rule.ensemble and ensemble_size is not None:
            raise ValueError(f"module {name!r} does not support ensemble members")
        if rule.clock and schedule is None and declaration.initial_time is None:
            raise ValueError(
                f"module {name!r} requires simulation_schedule or initial_time"
            )
    options = declaration.options
    for path, required in options.required_modules().items():
        missing = sorted(set(required).difference(opened))
        if missing:
            raise ValueError(
                f"option {path}={options.choice(path)!r} requires opened_modules to include {missing}"
            )

    device = declaration.device
    backend = resolve_backend(device)
    model_rule = spec.backend_requirements.get(
        backend.name, DEFAULT_BACKEND_REQUIREMENT
    )
    precision = declaration.precision
    mixed_precision = declaration.mixed_precision
    if mixed_precision is None:
        mixed_precision = model_rule.default_mixed_precision
        if mixed_precision is None:
            mixed_precision = backend.default_mixed_precision(device)
    if (
        device.type == "xpu"
        and (precision == "float64" or mixed_precision)
        and not float64_supported(device)
    ):
        raise ValueError(
            f"XPU device {str(device)!r} does not support FP64, but the model requests float64 storage through precision or mixed_precision"
        )
    initial_time = None
    if schedule is not None:
        if declaration.initial_time is not None:
            raise ValueError(
                "initial_time must not be configured together with simulation_schedule"
            )
        calendar = schedule.calendar
    else:
        calendar, normalized, _defaulted = normalize_calendar_dates(
            {"model initial_time": declaration.initial_time}, calendar=None
        )
        initial_time = normalized["model initial_time"]
    emulation = declaration.metal_emulation
    if emulation == "auto":
        emulation = (
            "float32x2" if backend.name == "metal" and mixed_precision else "native"
        )
    if emulation != "native":
        if backend.math.physics_fast_math:
            raise ValueError("Metal emulation requires HYDROFORGE_FAST_MATH=0")
        if (
            backend.name != "metal"
            or precision != "float32"
            or mixed_precision is not True
        ):
            raise ValueError(
                "metal_emulation requires Metal, float32 compute and mixed_precision=True"
            )
    backend.validate_precision(precision, mixed_precision and emulation == "native")
    model_rule._validate_precision(precision, mixed_precision, backend=backend.name)
    options.validate_backend(backend.name)
    block_size = declaration.block_size
    if block_size is None:
        block_size = model_rule.default_block_size
    if block_size is not None or backend.block.fixed is not None:
        block_size = resolve_block_size(backend, model_rule, block_size)
    if not model_rule.ensemble and ensemble_size is not None:
        raise ValueError(f"backend {backend.name!r} does not support ensemble members")

    topology = ProcessTopology.capture()
    return Selection(
        model=f"{spec.model_type.__module__}.{spec.model_type.__qualname__}",
        spec=spec,
        modules=opened,
        module_order=_module_order(spec, opened),
        options=options,
        conditions=conditions,
        backend=backend,
        device=device,
        precision=precision,
        dtype=torch.float32 if precision == "float32" else torch.float64,
        mixed_precision=mixed_precision,
        metal_emulation=emulation,
        backend_requirement=model_rule,
        block_size=block_size,
        capture=declaration.execution_mode == "auto" and model_rule.capture,
        calendar=calendar,
        initial_time=initial_time,
        schedule=schedule,
        rank=topology.rank,
        world_size=topology.world_size,
        spatial_rank=(topology.rank if parallel is None else parallel.spatial_rank),
        spatial_world_size=(
            topology.world_size if parallel is None else parallel.spatial_partitions
        ),
        ensemble_size=ensemble_size,
        local_ensemble_size=(
            ensemble_size if parallel is None else parallel.local_ensemble_size
        ),
        member_ids=None if parallel is None else parallel.member_ids,
        parallel=parallel,
    )

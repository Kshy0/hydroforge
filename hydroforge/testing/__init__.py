"""Test helpers that construct declarations outside a model runtime."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from types import MappingProxyType
from typing import Any, TypeVar

from pydantic import ConfigDict, InstanceOf, validate_call

from hydroforge.compiler.fields import bind_module_fields
from hydroforge.contracts.conditions import resolve_conditions
from hydroforge.contracts.fields import FieldDemandPlan
from hydroforge.contracts.options import OptionsConfig
from hydroforge.core.events import EventSink, NullEventSink
from hydroforge.declare.module import AbstractModule, construct_module
from hydroforge.declare.spec import ModuleBinding
from hydroforge.kernels.registry import KernelCall, Launch, intercepting

ModuleT = TypeVar("ModuleT", bound=AbstractModule)


@validate_call(config=ConfigDict(arbitrary_types_allowed=True, strict=True))
def build_module(
    module_type: type[ModuleT],
    values: Mapping[str, Any],
    *,
    references: tuple[InstanceOf[AbstractModule], ...] = (),
    outputs: frozenset[str] = frozenset(),
    batched_forcing: frozenset[str] = frozenset(),
    event_sink: InstanceOf[EventSink] | None = None,
    options: InstanceOf[OptionsConfig] | None = None,
) -> ModuleT:
    """Construct one module exactly as a model does, from explicit inputs.

    ``values`` are the module's field values; ``opened_modules`` defaults to
    this module plus its ``references``, which supply already constructed
    sibling modules. ``outputs`` names fields of this module requested
    directly by output, and ``batched_forcing`` its member-batched forcing.
    """

    values = dict(values)
    siblings = {module.module_name: module for module in references}
    if len(siblings) != len(references):
        raise ValueError("references contain duplicate module names")
    spec = module_type.spec()
    opened = values.setdefault("opened_modules", (spec.name, *siblings))
    if (
        type(opened) is not tuple
        or not opened
        or len(set(opened)) != len(opened)
        or spec.name not in opened
    ):
        raise ValueError("opened_modules must be a unique tuple including this module")
    for name, reference in spec.references.items():
        sibling = siblings.get(name)
        if name not in opened:
            if sibling is not None:
                raise ValueError(f"reference {name!r} supplies a closed module")
        elif sibling is None:
            raise ValueError(
                f"opened module reference {name!r} has no supplied instance"
            )
        reference.validate_type(type(sibling) if sibling is not None else None)
    for name in outputs:
        field = spec.tensor_fields.get(name)
        if field is None or field.excluded or field.tensor.output == "disabled":
            raise ValueError(f"output {name!r} is not an output-enabled tensor field")
    if batched_forcing and values.get("ensemble_size") is None:
        raise ValueError("batched_forcing requires ensemble_size")
    for name in batched_forcing:
        field = spec.tensor_fields.get(name)
        if field is None or field.tensor.category != "forcing":
            raise ValueError(f"batched_forcing {name!r} must name a forcing field")
    demand = FieldDemandPlan({spec.name: outputs}, {spec.name: outputs})
    conditions = resolve_conditions(
        (spec, *(type(module).spec() for module in siblings.values())), options
    )
    plan = bind_module_fields(
        spec,
        opened,
        demand,
        batched_forcing,
        conditions=conditions,
        module_specs={
            spec.name: spec,
            **{name: type(module).spec() for name, module in siblings.items()},
        },
    )
    if not (outputs | batched_forcing) <= plan.active:
        raise ValueError("requested output or batched forcing field is inactive")
    binding = ModuleBinding(
        plan=plan,
        references=MappingProxyType(
            {name: siblings.get(name) for name in spec.references}
        ),
        event_sink=NullEventSink() if event_sink is None else event_sink,
    )
    return construct_module(module_type, values, binding)


@contextmanager
def intercept_kernels(
    wrap: Callable[[KernelCall, Launch], Launch],
) -> Iterator[None]:
    """Wrap every kernel launch compiled inside the block.

    ``wrap(call, launch)`` returns the launch to use; it may filter on
    ``call.implementation.spec.name`` and read ``call.arguments``.  This is a
    hook for tests and benchmarks, not for model code.
    """

    with intercepting(wrap):
        yield


__all__ = ["build_module", "intercept_kernels"]

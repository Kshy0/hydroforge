# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""A model's structure as one self-contained, interactive HTML page.

:func:`model_structure` reads the compiled plan, so it describes any model:
its modules and their references, every field of the opened modules
(category, shape, dtype, coordinate, conditions), the coordinate ownership
graph rooted at the partition key, and the declared outputs; and the whole
declaration of the model class: closed modules, inactive fields and every
option with its choices, conflicts and gates. A materialized model adds what
only the runtime knows: resolved dimensions, the shape, dtype, device and
size of each resident tensor, the partition balance, halo and all-reduce
transports and, once its managed steps have run, every kernel they launch
with the fields each reads and writes, in recorded order.

:func:`render_html` embeds the document in a page that needs no network
access. Its map draws the modules (with the option choices that open them)
as a layered graph and the kernels, in launch order, across one lane per
module, marking what each reads and writes; a switch shows the current model
or the full declaration, and every module, kernel, field and option opens
its details. :meth:`AbstractModel.export_html` writes it.
"""

from __future__ import annotations

import html
import json
from collections.abc import Iterable, Mapping
from enum import Enum
from typing import TYPE_CHECKING, Any

import torch

from hydroforge.compiler.partition import bare

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel

_READS = frozenset({"read"})
_WRITES = frozenset({"write"})


def _text(value: Any) -> Any:
    """A JSON value of a declaration: conditions and types spelled as text."""

    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (tuple, list)):
        return [_text(item) for item in value]
    if isinstance(value, Mapping):
        return {str(key): _text(item) for key, item in value.items()}
    if isinstance(value, type):
        return f"{value.__module__}.{value.__qualname__}"
    return str(value)


def _selector(selector: Any) -> str:
    """A view selector as ``range(start, stop, step)`` or ``[i, j, ...]``."""

    values = getattr(selector, "values", None)
    if values is not None:
        return str(list(values))
    stop = "end" if selector.stop is None else selector.stop
    step = "" if selector.step == 1 else f", {selector.step}"
    return f"range({selector.start}, {stop}{step})"


def _peek(owner: Any, name: str) -> Any:
    """A materialized attribute, without evaluating lazy virtual fields."""

    value = owner.__dict__.get(name)
    if value is None:
        value = owner.__dict__.get(f"__derived_reference_index_{name}")
    return value


def _layers(nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> dict[str, int]:
    """Depth of every node below its sources (edges point to the source)."""

    nodes = list(nodes)
    parents: dict[str, list[str]] = {node: [] for node in nodes}
    for child, parent in edges:
        if child in parents and parent in parents and child != parent:
            parents[child].append(parent)
    depth: dict[str, int] = {}

    def visit(node: str, trail: frozenset[str]) -> int:
        if node in depth:
            return depth[node]
        known = [visit(p, trail | {node}) for p in parents[node] if p not in trail]
        depth[node] = 1 + max(known) if known else 0
        return depth[node]

    for node in nodes:
        visit(node, frozenset())
    return depth


def _tensor(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, torch.Tensor):
        return None
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype).removeprefix("torch."),
        "device": str(value.device),
        "bytes": value.numel() * value.element_size(),
    }


class _Names:
    """Field names of resident tensors, by identity."""

    def __init__(self, runtime: Any) -> None:
        self.names: dict[int, str] = {}
        if runtime is None:
            return
        plan = runtime.plan
        for entry in plan.fields.entries.values():
            module = runtime.modules.get(entry.module)
            if module is None:
                continue
            value = _peek(module, entry.name)
            if isinstance(value, torch.Tensor):
                self.names.setdefault(id(value), entry.qualified)
        halo = getattr(runtime, "halo", None)
        if halo is not None:
            for tensor in halo.tensors():
                self.names.setdefault(id(tensor), "halo buffer")
        execution = runtime.execution
        if execution is not None:
            for reducer in execution.reducers.values():
                for tensor in reducer.tensors():
                    self.names.setdefault(id(tensor), "reduce buffer")
            for (source, _dtype), tensor in execution.step_fields.buffers.items():
                self.names.setdefault(id(tensor), f"step.{source}")

    def add(self, tensor: Any, name: str) -> None:
        if isinstance(tensor, torch.Tensor):
            self.names.setdefault(id(tensor), name)

    def __call__(self, tensors: Iterable[Any]) -> list[str]:
        names = []
        for tensor in tensors:
            if not isinstance(tensor, torch.Tensor):
                continue
            name = self.names.get(id(tensor))
            if name is None:
                name = f"temporary{list(tensor.shape)}"
            if name not in names:
                names.append(name)
        return names


def _implementation(implementation: Any) -> str:
    """``Triton``, ``Cuda``, ``Metal``, ``Torch``: the implementation's kind."""

    name = type(implementation).__name__.lstrip("_")
    for suffix in ("KernelImplementation", "Implementation"):
        name = name.removesuffix(suffix)
    return name


def _kernel_call(call: Any, reads: Any, writes: Any, names: _Names) -> dict:
    implementation = call.implementation
    spec = implementation.spec
    if reads is None:
        reads, writes = [], []
        for name, mode in spec.buffers.items():
            value = call.arguments.get(name)
            if mode not in _WRITES:
                reads.append(value)
            if mode not in _READS:
                writes.append(value)
    return {
        "kind": "kernel",
        "name": spec.name,
        "implementation": _implementation(implementation),
        "reads": names(reads),
        "writes": names(writes),
    }


def _operators(program: Any, names: _Names) -> list[dict]:
    """Every operator of one recorded program, nested loops flattened."""

    result = []
    for operator in getattr(program, "operators", ()):
        call = getattr(operator, "call", None)
        kind = type(operator).__name__
        if call is not None and hasattr(call, "implementation"):
            result.append(_kernel_call(call, operator.reads, operator.writes, names))
        elif kind == "TorchOperator":
            arguments, keywords, _outputs = operator.static_values()
            inputs = [
                *(arguments if isinstance(arguments, (tuple, list)) else ()),
                *(keywords.values() if isinstance(keywords, Mapping) else ()),
            ]
            function = getattr(operator.function, "__name__", str(operator.function))
            result.append(
                {
                    "kind": "torch",
                    "name": str(function),
                    "implementation": "aten",
                    "reads": names(inputs),
                    "writes": names(operator.writes),
                }
            )
        elif kind == "CollectiveOperator":
            result.append(
                {
                    "kind": "collective",
                    "name": f"{operator.operation}({operator.reduction})",
                    "implementation": operator.scope,
                    "reads": names(operator.reads),
                    "writes": names(operator.writes),
                }
            )
        elif kind == "PredicateLoopOperator":
            result.append(
                {
                    "kind": "loop",
                    "name": "predicate loop",
                    "implementation": "",
                    "reads": [],
                    "writes": [],
                    "body": _operators(operator.loop.body, names),
                }
            )
        else:
            result.append(
                {
                    "kind": kind,
                    "name": kind,
                    "implementation": "",
                    "reads": names(getattr(operator, "reads", ())),
                    "writes": names(getattr(operator, "writes", ())),
                }
            )
    return result


def _steps(runtime: Any, names: _Names) -> list[dict]:
    """Managed steps with their eager and recorded kernels."""

    execution = runtime.execution
    steps: dict[str, dict] = {}
    for key, loop in execution.programs.items():
        owner = key[0] if isinstance(key, tuple) and key else None
        function = getattr(owner, "function", None)
        step = getattr(function, "__name__", str(owner))
        kind = key[1] if isinstance(key, tuple) and len(key) > 1 else "program"
        entry = steps.setdefault(step, {"name": step, "programs": []})
        frame = getattr(loop, "frame", None)
        if frame is not None:
            names.add(frame.index, "substep.index")
            names.add(frame.dt, "substep.dt")
        for control in ("counter", "continue_flag", "count", "weight"):
            names.add(getattr(loop, control, None), f"loop.{control}")
        roles = []
        if hasattr(loop, "operators"):
            roles.append(("body", loop))
        else:
            for role in ("proposal", "body", "final"):
                program = getattr(loop, role, None)
                if program is not None:
                    roles.append((role, program))
        entry["programs"].append(
            {
                "kind": str(kind),
                "roles": [
                    {"role": role, "operators": _operators(program, names)}
                    for role, program in roles
                ],
            }
        )
    # Kernels launched outside recorded scopes with model-bound arguments
    # (initialization and step bodies), in first-launch order.
    eager = []
    binder = execution.kernel_binding
    bound = getattr(binder, "_complete_cache", {})
    for key, (registry, _launch) in getattr(binder, "_launch_cache", {}).items():
        spec = registry.spec_for(runtime.plan.backend)
        arguments = bound[key][1] if key in bound else {}
        reads, writes = [], []
        for name, mode in spec.buffers.items():
            value = arguments.get(name)
            if mode not in _WRITES:
                reads.append(value)
            if mode not in _READS:
                writes.append(value)
        implementation = registry.implementation(
            runtime.plan.backend, precision=runtime.plan.precision
        )
        eager.append(
            {
                "kind": "kernel",
                "name": spec.name,
                "implementation": _implementation(implementation),
                "reads": names(reads),
                "writes": names(writes),
            }
        )
    return [*steps.values(), *([{"name": "(eager)", "eager": eager}] if eager else [])]


def _holds(condition: Any, context: Any) -> bool | None:
    """Whether a condition holds now; ``None`` when it cannot be decided."""

    try:
        return bool(condition.evaluate(context))
    except Exception:  # noqa: BLE001 - shown as undecided
        return None


def _condition(condition: Any, context: Any) -> dict[str, Any] | None:
    if condition is None:
        return None
    return {"text": str(condition), "holds": _holds(condition, context)}


_BOUNDS = {"gt": ">", "ge": "≥", "lt": "<", "le": "≤"}


def _constraints(field: Any) -> list[str]:
    """Pydantic constraints spelled ``> 0``, ``≥ 2``, ``multiple of 2``."""

    spelled = []
    for item in getattr(field, "metadata", ()) or ():
        for name, symbol in _BOUNDS.items():
            if hasattr(item, name):
                spelled.append(f"{symbol} {getattr(item, name)}")
                break
        else:
            if hasattr(item, "multiple_of"):
                spelled.append(f"multiple of {item.multiple_of}")
            elif hasattr(item, "allow_inf_nan"):
                spelled.append(
                    "finite" if not item.allow_inf_nan else "inf/nan allowed"
                )
            else:
                spelled.append(str(item))
    return spelled


def _type(annotation: Any) -> str:
    """An annotation without ``builtins.``/``typing.`` qualifiers."""

    return _text(annotation).replace("builtins.", "").replace("typing.", "")


def _options(model: AbstractModel, context: Any) -> list[dict[str, Any]]:
    """Every option of the model's options record, nested records first."""

    from hydroforge.contracts.options import OptionsConfig
    from hydroforge.core.validation import field_default

    entries: list[dict[str, Any]] = []

    def setting(path: str, value: Any) -> str:
        nested: Any = value
        for part in reversed(path.split(".")):
            nested = {part: nested}
        return f"options={json.dumps(nested, default=str)}"

    def visit(record: Any, prefix: str) -> None:
        own = getattr(type(record), "__relevant_when__", None)
        for name, field in type(record).model_fields.items():
            path = f"{prefix}.{name}" if prefix else name
            value = getattr(record, name)
            if isinstance(value, OptionsConfig):
                relevance = getattr(type(value), "__relevant_when__", None)
                entries.append(
                    {
                        "path": path,
                        "kind": "record",
                        "type": _text(type(value)),
                        "description": field.description
                        or next(
                            iter((type(value).__doc__ or "").strip().splitlines()), ""
                        ),
                        "relevant_when": None
                        if relevance is None
                        else _condition(
                            _as_condition(relevance).prefixed(path), context
                        ),
                    }
                )
                visit(value, path)
                continue
            metadata = OptionsConfig._field_metadata(field) or {}
            role = metadata.get("role", "value")
            default = field_default(field)
            current = value.value if isinstance(value, Enum) else value
            relevance = [
                item.prefixed(prefix) if prefix else item
                for item in (
                    None if own is None else _as_condition(own),
                    metadata.get("relevant_when"),
                )
                if item is not None
            ]
            effect = metadata.get("affects_parameters", False)
            entry: dict[str, Any] = {
                "path": path,
                "kind": {"option": "choice", "forcing": "forcing"}.get(role, "value"),
                "type": _type(field.annotation),
                "default": _text(default),
                "value": _text(current),
                "changed": _text(current) != _text(default),
                "description": field.description,
                "constraints": _constraints(field),
                "relevant_when": [_condition(item, context) for item in relevance],
                "affects_parameters": effect
                if isinstance(effect, bool)
                else _condition(effect.prefixed(prefix) if prefix else effect, context),
                "set": setting(path, _text(current)),
            }
            conflicts = metadata.get("conflicts") or {}
            codes = metadata.get("codes")
            if codes is not None:
                requires = (
                    metadata.get("requires_modules") or metadata.get("requires") or {}
                )
                traits = metadata.get("traits") or {}
                backends = metadata.get("supported_backends") or {}
                entry["on_missing"] = metadata.get("on_missing")
                entry["choices"] = [
                    {
                        "choice": choice,
                        "code": code,
                        "selected": str(choice) == str(entry["value"]),
                        "requires": list(requires.get(choice, ())),
                        "traits": _text(traits.get(choice, {})),
                        "backends": sorted(
                            backend
                            for backend, allowed in backends.items()
                            if choice in allowed
                        )
                        if backends
                        else None,
                        "conflicts": [
                            {
                                **_condition(
                                    condition.prefixed(prefix) if prefix else condition,
                                    context,
                                ),
                                "message": message,
                            }
                            for condition, message in (
                                conflicts.get(choice, ())
                                if isinstance(conflicts, Mapping)
                                else ()
                            )
                        ],
                    }
                    for choice, code in codes.items()
                ]
            elif conflicts:
                entry["conflicts"] = [
                    {
                        **_condition(
                            condition.prefixed(prefix) if prefix else condition, context
                        ),
                        "message": message,
                    }
                    for condition, message in conflicts
                ]
            entries.append(entry)

    visit(model.options, "")
    return entries


def _as_condition(value: Any) -> Any:
    from hydroforge.contracts.conditions import as_condition

    return as_condition(value)


def _declaration(model: AbstractModel) -> dict[str, Any]:
    """Everything the model class declares, evaluated against its options.

    Unlike the plan, this covers closed modules and inactive fields too, so
    the model's logic reads off one page: every option with its type,
    default, choices, codes, requirements, conflicts and relevance, which
    fields and modules each option gates, and every field with its default,
    constraints and the conditions that make it active.
    """

    from hydroforge.contracts.conditions import ConditionContext
    from hydroforge.execution.step import _ManagedStepDescriptor

    plan = model.plan
    spec = plan.spec
    model_type = type(model)
    context = ConditionContext(frozenset(plan.modules), options=model.options)
    options = _options(model, context)
    users: dict[str, list[str]] = {}

    def gate(label: str, conditions: Any) -> list[dict[str, Any]]:
        result = []
        for condition in conditions or ():
            condition = _as_condition(condition)
            for path in condition.option_paths():
                users.setdefault(path, []).append(label)
            result.append(_condition(condition, context))
        return result

    modules = []
    fields = []
    for name, module_spec in spec.modules.items():
        module_type = module_spec.module_type
        pydantic_fields = getattr(module_type, "model_fields", {})
        for field_name, field_spec in module_spec.tensor_fields.items():
            tensor = field_spec.tensor
            declared = pydantic_fields.get(field_name)
            default = None
            if declared is not None and not field_spec.computed:
                from pydantic_core import PydanticUndefined

                raw = declared.default
                default = (
                    "required"
                    if raw is PydanticUndefined and declared.default_factory is None
                    else _text(raw)
                )
            entry = plan.fields.entries.get(f"{name}.{field_name}")
            label = f"{name}.{field_name}"
            fields.append(
                {
                    "module": name,
                    "name": field_name,
                    "category": tensor.category,
                    "dtype": tensor.dtype,
                    "shape": [_text(token) for token in tensor.shape],
                    "computed": field_spec.computed,
                    "default": default,
                    "units": tensor.units,
                    "bounds": [
                        f"{key} {value}"
                        for key, value in (
                            ("≥", tensor.ge),
                            (">", tensor.gt),
                            ("≤", tensor.le),
                            ("<", tensor.lt),
                        )
                        if value is not None
                    ]
                    + (["finite"] if tensor.finite else []),
                    "depends_on": gate(label, tensor.depends_on),
                    "required_by": gate(label + " (required by)", tensor.required_by),
                    "materialize": gate(label + " (materialize)", tensor.materialize),
                    "expression": tensor.expression,
                    "output": tensor.output,
                    "active": None if entry is None else entry.active,
                    "description": field_spec.description,
                }
            )
        requirement = spec.module_requirements.get(name)
        modules.append(
            {
                "name": name,
                "class": _text(module_type),
                "opened": name in plan.modules,
                "required": name in spec.required_modules,
                "references": [
                    ref.name + ("" if not ref.optional else "?")
                    for ref in module_spec.references.values()
                ],
                "conflicts": list(module_spec.conflicts),
                "requirement": None
                if requirement is None
                else requirement.model_dump(mode="json"),
                "opened_by_options": [
                    f"{item.path}={item.choice} ({item.on_missing})"
                    for item in model.options.option_requirements()
                    if name in item.modules
                ],
                "fields": len(module_spec.tensor_fields),
                "description": module_spec.description,
            }
        )
    for entry in options:
        entry["used_by"] = sorted(set(users.get(entry["path"], ())))
    steps = [
        {
            "name": attribute,
            "doc": (value.function.__doc__ or "").strip(),
        }
        for attribute, value in vars(model_type).items()
        if isinstance(value, _ManagedStepDescriptor)
    ]
    initialize = model_type.__dict__.get("initialize_model_state")
    return {
        "options_class": _text(type(model.options)),
        "options": options,
        "modules": modules,
        "fields": fields,
        "backends": {
            name: requirement.model_dump(mode="json")
            for name, requirement in spec.backend_requirements.items()
        },
        "steps": steps,
        "initialize": None
        if initialize is None
        else (initialize.__doc__ or "").strip(),
        "step_fields": {
            "providers": sorted(spec.step_fields.providers),
            "expressions": {
                name: str(expression)
                for name, expression in spec.step_fields.expressions.items()
            },
        },
        "partition": {"key": spec.partition_key, "group": spec.partition_group},
        "grid_axes": {
            symbol: axis.model_dump() for symbol, axis in spec.grid_axes.items()
        },
        "kernel_fields": list(spec.kernel_fields),
    }


def model_structure(model: AbstractModel) -> dict[str, Any]:
    """The JSON-ready structure document of ``model``.

    It holds the compiled plan and runtime (the current model) and the
    model's complete declaration (see :func:`_declaration`): closed modules,
    inactive fields and every option with its defaults, choices, conflicts
    and what it gates.
    """

    from hydroforge.execution.session import ModelRuntime

    plan = model.plan
    spec = plan.spec
    runtime = ModelRuntime.of(model)
    materialized = runtime.state == "materialized" and runtime.execution is not None
    live = runtime if materialized else None
    names = _Names(live)
    model_type = type(model)

    requirements = spec.module_requirements
    modules = []
    for name, module_spec in spec.modules.items():
        requirement = requirements.get(name)
        modules.append(
            {
                "name": name,
                "class": _text(module_spec.module_type),
                "description": module_spec.description,
                "opened": name in plan.modules,
                "required": name in spec.required_modules,
                "references": [
                    {"name": ref.name, "optional": ref.optional}
                    for ref in module_spec.references.values()
                ],
                "conflicts": list(module_spec.conflicts),
                "requirement": None
                if requirement is None
                else requirement.model_dump(mode="json"),
                "kernel_fields": list(module_spec.kernel_fields),
            }
        )
    module_edges = [
        (module["name"], ref["name"])
        for module in modules
        for ref in module["references"]
    ]
    depth = _layers((m["name"] for m in modules), module_edges)
    for module in modules:
        module["depth"] = depth[module["name"]]

    dimensions: dict[str, int] = {}
    fields = []
    for entry in plan.fields.entries.values():
        tensor = entry.tensor
        value = None
        if live is not None and entry.module in live.modules:
            value = _peek(live.modules[entry.module], entry.name)
            if isinstance(value, torch.Tensor):
                logical = list(value.shape)[value.ndim - len(tensor.shape) :]
                for token, extent in zip(tensor.shape, logical, strict=False):
                    if isinstance(token, str):
                        dimensions.setdefault(bare(token), int(extent))
        fields.append(
            {
                "module": entry.module,
                "name": entry.name,
                "active": entry.active,
                "computed": entry.computed,
                "reference_index": entry.reference_index,
                "category": tensor.category,
                "dtype": tensor.dtype,
                "shape": [_text(token) for token in tensor.shape],
                "dim_coords": tensor.dim_coords,
                "coordinate": tensor.is_coordinate,
                "key": tensor.is_key,
                "partition_by": tensor.partition_by,
                "references": tensor.references,
                "selects": tensor.selects,
                "output": tensor.output,
                "mode": tensor.mode,
                "units": tensor.units,
                "expression": tensor.expression,
                "depends_on": _text(tensor.depends_on),
                "after_init": tensor.after_init,
                "description": entry.description,
                "runtime": _tensor(value),
            }
        )

    schema = plan.fields.partition
    coordinates = []
    coordinate_edges = []
    for name in sorted(schema.coordinates):
        metadata = schema.fields[name]
        via = bare(metadata.partition_by)
        target = (
            bare(schema.fields[via].references) if via else bare(metadata.references)
        )
        owner = next(
            (f["module"] for f in fields if f["name"] == name and f["coordinate"]), None
        )
        coordinates.append(
            {
                "name": name,
                "module": owner,
                "via": via or (name if target else None),
                "target": target,
                "replicated": metadata.replicated,
                "selection": schema.selections.get(name),
                "extent": dimensions.get(name),
            }
        )
        if target:
            coordinate_edges.append((name, target))
    depth = _layers((c["name"] for c in coordinates), coordinate_edges)
    for coordinate in coordinates:
        coordinate["depth"] = depth[coordinate["name"]]

    partition = {
        "key": spec.partition_key,
        "group": spec.partition_group,
        "spatial_rank": plan.spatial_rank,
        "spatial_world_size": plan.spatial_world_size,
        "imbalance_tolerance": plan.partition_imbalance,
        "balance": None,
        "halo": None,
        "reduce": {},
    }
    partition_runtime = None if live is None else getattr(live, "partition", None)
    if (
        partition_runtime is not None
        and plan.spatial_world_size > 1
        and spec.partition_key
    ):
        pieces = partition_runtime.pieces
        partition["balance"] = {
            "split": pieces.split,
            "pieces": pieces.pieces,
            "imbalance": round(pieces.imbalance, 4),
            "cut": pieces.cut,
        }
    execution = None if live is None else getattr(live, "execution", None)
    for name, reducer in getattr(execution, "reducers", {}).items():
        partition["reduce"][name] = {
            "transports": list(reducer.transports),
            "skipped": dict(reducer.skipped),
        }
    if live is not None and getattr(live, "halo", None) is not None:
        halo = live.halo
        counts = halo.plan.counts
        rank = halo.plan.rank
        partition["halo"] = {
            "owned": halo.plan.n_owned,
            "ghosts": halo.plan.n_ghosts,
            "ghosts_total": halo.plan.total_ghosts,
            "transport": halo.transport,
            "skipped": dict(halo.skipped),
            "direct": halo.direct,
            "sends": {
                str(p): int(counts[rank, p])
                for p in range(halo.plan.world)
                if counts[rank, p]
            },
            "receives": {
                str(q): int(counts[q, rank])
                for q in range(halo.plan.world)
                if counts[q, rank]
            },
        }

    output = plan.output
    config = output.config
    statistics_declaration = output.declaration
    outputs = {
        "directory": None if output.directory is None else str(output.directory),
        "sink": config.sink,
        "windows": None if output.windows is None else _text(output.windows),
        "views": {
            name: {axis: _selector(selector) for axis, selector in view.axes.items()}
            for name, view in config.views.items()
        },
        "variables": [
            {
                "operation": operation,
                "name": item.name,
                "expression": item.expression,
                "view": item.view,
                "axes": None
                if statistics_declaration is None
                else list(statistics_declaration.axes.get(item.name, ())),
            }
            for operation, items in config.variables.items()
            for item in items
        ],
    }
    statistics = getattr(live, "statistics", None) if live is not None else None
    layouts = getattr(statistics, "_statistics_layouts", None)
    if layouts:
        for variable in outputs["variables"]:
            layout = layouts.get(variable["name"])
            variable["shape"] = None if layout is None else list(layout.actual_shape)

    try:
        options = plan.options.model_dump(mode="json")
    except Exception:  # noqa: BLE001 - shown as text instead
        options = str(plan.options)

    overview = {
        "model": f"{model_type.__module__}.{model_type.__qualname__}",
        "state": runtime.state,
        "backend": plan.backend.name,
        "device": str(plan.device),
        "precision": plan.precision,
        "mixed_precision": plan.mixed_precision,
        "ensemble_size": plan.ensemble_size,
        "opened_modules": list(plan.modules),
        "module_order": list(plan.module_order),
        "execution": None
        if live is None
        else getattr(
            live.execution.executor, "name", type(live.execution.executor).__name__
        ),
        "missing_inputs": sorted(getattr(live, "missing_inputs", ()) or ()),
        "grid_axes": {
            symbol: _text(axis.model_dump()) for symbol, axis in spec.grid_axes.items()
        },
    }
    return {
        "title": model_type.__qualname__,
        "declaration": _declaration(model),
        "overview": overview,
        "options": options,
        "conditions": _text(dict(plan.conditions)),
        "modules": modules,
        "fields": fields,
        "dimensions": dimensions,
        "coordinates": coordinates,
        "partition": partition,
        "outputs": outputs,
        "steps": [] if live is None else _steps(live, names),
    }


def render_html(document: Mapping[str, Any]) -> str:
    """One self-contained page showing a :func:`model_structure` document."""

    payload = json.dumps(document, default=str).replace("</", "<\\/")
    return _TEMPLATE.replace(
        "{{TITLE}}", html.escape(str(document.get("title", "model")))
    ).replace("{{DATA}}", payload)


_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{{TITLE}} — HydroForge model structure</title>
<style>
:root{--bg:#f5f6f9;--fg:#1d2330;--muted:#6b7487;--line:#dde1ea;--card:#fff;--accent:#2f6fdf;--soft:#e7eefc;
--r:#2f8f5b;--w:#c2453d;--rw:#8a55c7;--chip:#eef1f7;--warn:#d08a1e;--lane:#f3f5f9;--opt:#b7791f}
@media (prefers-color-scheme:dark){:root{--bg:#121519;--fg:#e4e7ee;--muted:#98a1b3;--line:#2c323d;--card:#1a1e26;
--accent:#6b9cff;--soft:#1f2a40;--chip:#252b36;--lane:#171b22;--opt:#e0a84a}}
*{box-sizing:border-box}body{margin:0;font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif;background:var(--bg);color:var(--fg)}
header{position:sticky;top:0;z-index:5;background:var(--card);border-bottom:1px solid var(--line);padding:10px 20px;
display:flex;gap:14px;align-items:center;flex-wrap:wrap}header h1{font-size:18px;margin:0}
.switch{display:inline-flex;border:1px solid var(--line);border-radius:8px;overflow:hidden}
.switch button{border:0;background:var(--card);color:var(--fg);padding:5px 12px;cursor:pointer;font:inherit}
.switch button.on{background:var(--accent);color:#fff}
nav{display:flex;gap:2px;padding:0 20px;background:var(--card);border-bottom:1px solid var(--line)}
nav button{border:0;border-bottom:2px solid transparent;background:none;color:var(--muted);padding:8px 12px;cursor:pointer;font:inherit}
nav button.on{color:var(--fg);border-bottom-color:var(--accent)}
.shell{display:grid;grid-template-columns:minmax(0,1fr);gap:16px;padding:16px 20px}
.shell.open{grid-template-columns:minmax(0,1fr) 380px}
aside{position:sticky;top:110px;align-self:start;max-height:calc(100vh - 130px);overflow:auto;background:var(--card);
border:1px solid var(--line);border-radius:10px;padding:12px 14px}aside h2{margin-top:0}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:12px 14px;margin:0 0 16px;overflow-x:auto}
h2{font-size:15px;margin:2px 0 10px}h3{font-size:13px;margin:12px 0 4px;color:var(--muted);font-weight:600}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{border-bottom:1px solid var(--line);padding:4px 7px;text-align:left;vertical-align:top}
th{position:sticky;top:0;background:var(--card);cursor:pointer;user-select:none;white-space:nowrap}
tr.off td{color:var(--muted)}tr.click{cursor:pointer}tr.click:hover td{background:var(--soft)}
code,.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px}
.chip{display:inline-block;background:var(--chip);border-radius:10px;padding:0 7px;margin:1px 2px;font-size:12px}
.chip.link{cursor:pointer}.chip.link:hover{outline:1px solid var(--accent)}
.muted{color:var(--muted)}.controls{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:10px;align-items:center}
input,select{background:var(--card);color:var(--fg);border:1px solid var(--line);border-radius:6px;padding:4px 8px;font:inherit}
.kv{display:grid;grid-template-columns:max-content 1fr;gap:3px 16px}.kv>div:nth-child(odd){color:var(--muted)}.kv>div{overflow-wrap:anywhere}
.legend{display:flex;gap:14px;flex-wrap:wrap;font-size:12px;color:var(--muted);margin-top:8px}.legend i{display:inline-block;width:12px;height:12px;border-radius:3px;vertical-align:-2px;margin-right:4px}
.R{color:var(--r)}.W{color:var(--w)}.RW{color:var(--rw)}pre{white-space:pre-wrap;margin:0}details summary{cursor:pointer}
svg{display:block}svg text{fill:var(--fg);font-size:12px}svg .muted{fill:var(--muted)}
.node rect{fill:var(--card);stroke:var(--line);stroke-width:1.2}.node{cursor:pointer}.node:hover rect{stroke:var(--accent)}
.node.closed rect{stroke-dasharray:5 3;fill:var(--lane)}.node.closed text{fill:var(--muted)}
.node.option rect{stroke:var(--opt)}.node.sel rect{stroke:var(--accent);stroke-width:2.5}.node.dim{opacity:.25}
.node.key rect{stroke:var(--accent);stroke-width:2}
.edge{fill:none;stroke:var(--muted);stroke-width:1.2;opacity:.75}.edge.opt{stroke:var(--opt);stroke-dasharray:4 3}
.edge.conflict{stroke:var(--w);stroke-dasharray:2 3}.edge.hot{stroke:var(--accent);stroke-width:2.2;opacity:1}.edge.dim{opacity:.12}
.row{cursor:pointer}.row rect.bg{fill:transparent}.row:hover rect.bg{fill:var(--soft)}.row.sel rect.bg{fill:var(--soft)}.row.dim{opacity:.28}
.lanehead{cursor:pointer}.lanehead:hover text{fill:var(--accent)}
</style></head><body>
<header><h1 id="title"></h1><span class="muted" id="subtitle"></span><span style="flex:1"></span>
<span class="switch" id="mode"><button data-mode="current" title="Opened modules, active fields and the current option values">Current model</button><button data-mode="full" title="Everything the model class declares: closed modules, inactive fields, every option choice">Full declaration</button></span>
<input id="q" placeholder="search…" size="22"></header>
<nav id="nav"></nav>
<div class="shell" id="shell"><main id="main"></main><aside id="drawer" hidden></aside></div>
<script type="application/json" id="data">{{DATA}}</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent),X=D.declaration;
const S={mode:'current',tab:'map',sel:null,q:''};
const NS='http://www.w3.org/2000/svg';
const $=(t,a={},...c)=>{const e=document.createElement(t);for(const[k,v]of Object.entries(a)){if(v==null)continue;
 if(k.startsWith('on'))e.addEventListener(k.slice(2),v);else e.setAttribute(k,v)}
 for(const x of c.flat(9))if(x!=null&&x!==false)e.append(x instanceof Node?x:document.createTextNode(String(x)));return e};
const mk=(t,a={},text)=>{const e=document.createElementNS(NS,t);for(const[k,v]of Object.entries(a))if(v!=null)e.setAttribute(k,v);if(text!=null)e.textContent=text;return e};
const val=v=>v==null||v===''||(Array.isArray(v)&&!v.length)?'':typeof v==='object'?JSON.stringify(v):String(v);
const fmtBytes=b=>b==null?'':b<1024?b+' B':b<1048576?(b/1024).toFixed(1)+' KiB':b<1073741824?(b/1048576).toFixed(1)+' MiB':(b/1073741824).toFixed(2)+' GiB';
const fit=(x,n)=>x&&x.length>n?x.slice(0,n-1)+'…':(x||'');
const chips=xs=>(xs||[]).map(x=>$('span',{class:'chip'},x));
const card=(title,...body)=>$('div',{class:'card'},title?$('h2',{},title):null,body);
const full=()=>S.mode==='full';
const match=s=>!S.q||String(s).toLowerCase().includes(S.q);
function kv(obj){const g=$('div',{class:'kv'});for(const[k,v]of Object.entries(obj)){if(v==null||v===''||(Array.isArray(v)&&!v.length)||(typeof v==='object'&&!Array.isArray(v)&&!Object.keys(v).length))continue;
 g.append($('div',{},k),Array.isArray(v)&&v.every(x=>typeof x!=='object')?$('div',{},chips(v)):v instanceof Node?$('div',{},v):$('div',{class:'mono'},typeof v==='object'?JSON.stringify(v):String(v)))}return g}
function table(cols,rows,{rowClass,onrow}={}){let key=null,dir=1;const t=$('table'),tb=$('tbody');
 t.append($('thead',{},$('tr',{},cols.map(([l],i)=>$('th',{onclick:()=>{dir=key===i?-dir:1;key=i;draw()}},l)))),tb);
 function draw(){tb.innerHTML='';const rs=rows.slice();if(key!=null)rs.sort((a,b)=>val(cols[key][1](a,true)).localeCompare(val(cols[key][1](b,true)),undefined,{numeric:true})*dir);
  for(const r of rs){const tr=$('tr',{class:[rowClass?rowClass(r):'',onrow?'click':''].join(' ')},cols.map(([,f])=>{const v=f(r);return $('td',{},v instanceof Node||Array.isArray(v)?v:val(v))}));
   if(onrow)tr.addEventListener('click',e=>{if(!e.target.closest('.link'))onrow(r)});tb.append(tr)}}
 draw();return t}
const cond=c=>c?$('span',{class:'chip',title:c.holds===true?'holds now':c.holds===false?'does not hold now':'undecided',
 style:`border:1px solid ${c.holds===true?'var(--r)':c.holds===false?'var(--w)':'var(--line)'}`},(c.holds===true?'✓ ':c.holds===false?'✗ ':'? ')+c.text):'';
// A conflict binds only an active option: ⚠ marks what would conflict with the current settings.
const conflict=k=>$('div',{},$('span',{class:'chip',title:k.holds?'selecting this conflicts with the current settings':'no conflict with the current settings',
 style:`border:1px solid ${k.holds?'var(--warn)':'var(--line)'}`},(k.holds?'⚠ ':'')+'conflicts when '+k.text),k.message?$('div',{class:'muted'},k.message):null);

// ---------------------------------------------------------------- indexes
const order=D.overview.module_order;
const planMods=Object.fromEntries(D.modules.map(m=>[m.name,m]));
const declMods=Object.fromEntries(X.modules.map(m=>[m.name,m]));
const planFields=Object.fromEntries(D.fields.map(f=>[f.module+'.'+f.name,f]));
const declFields=Object.fromEntries(X.fields.map(f=>[f.module+'.'+f.name,f]));
const options=Object.fromEntries(X.options.map(o=>[o.path,o]));
const laneOf=n=>planFields[n]?planFields[n].module:/^(step|substep|loop)\./.test(n)?'step & loop':/buffer$/.test(n)?'communication':'temporaries';
const K=[];let nextLoop=0;
for(const step of D.steps){const blocks=step.eager?[['eager launches',step.eager]]:step.programs.flatMap(p=>p.roles.map(r=>[p.kind+' · '+r.role,r.operators]));
 for(const[label,ops]of blocks){K.push({head:true,step:step.name,label});
  const walk=(os,depth,loops)=>{for(const o of os){const k={...o,id:K.length,step:step.name,block:label,depth,loops};K.push(k);
   if(o.body){const id=nextLoop++;walk(o.body,depth+1,[...loops,id]);k.loopId=id}}};walk(ops,0,[])}}
const kernels=K.filter(k=>!k.head);
const touching=name=>kernels.filter(k=>k.reads.includes(name)||k.writes.includes(name));
const moduleKernels=m=>kernels.filter(k=>[...k.reads,...k.writes].some(n=>laneOf(n)===m));

// ---------------------------------------------------------------- selection
function select(type,id){S.sel=S.sel&&S.sel.type===type&&S.sel.id===id?null:{type,id};render()}
const link=(type,id,label)=>$('span',{class:'chip link',onclick:()=>select(type,id),title:'show '+type},label||id);
function drawer(){const d=document.getElementById('drawer'),shell=document.getElementById('shell');d.innerHTML='';
 if(!S.sel){d.hidden=true;shell.classList.remove('open');return}d.hidden=false;shell.classList.add('open');
 const close=$('span',{style:'float:right;cursor:pointer',class:'muted',onclick:()=>{S.sel=null;render()}},'✕');
 const{type,id}=S.sel;d.append(close);const add=(...xs)=>d.append($('div',{},xs));
 if(type==='module'&&!declMods[id]||type==='option'&&!options[id]||type==='field'&&!declFields[id]&&!planFields[id]){add($('div',{class:'muted'},'nothing named '+id));return}
 if(type==='module'){const m=declMods[id]||{},p=planMods[id]||{};
  const fields=X.fields.filter(f=>f.module===id&&(full()||f.active));
  add($('h2',{},'module ',id),$('div',{class:'muted'},m.description||p.description),
   kv({class:m.class,opened:m.opened?'yes':'no',required:m.required?'yes':'optional',references:m.references,'conflicts with':m.conflicts,
    'opened by options':m.opened_by_options,requirement:m.requirement,'kernel fields':p.kernel_fields}),
   $('h3',{},'kernels touching it ('+moduleKernels(id).length+')'),moduleKernels(id).map(k=>link('kernel',k.id,k.name)),
   $('h3',{},(full()?'declared':'active')+' fields ('+fields.length+')'),fields.map(f=>link('field',id+'.'+f.name,f.name)))}
 if(type==='kernel'){const k=K[id];const group=names=>{const g={};for(const n of names)(g[laneOf(n)]=g[laneOf(n)]||[]).push(n);return g};
  const part=(title,names)=>[$('h3',{},title+' ('+names.length+')'),Object.entries(group(names)).map(([lane,ns])=>$('div',{},$('span',{class:'muted'},lane+': '),
   ns.map(n=>planFields[n]?link('field',n,n.split('.').slice(1).join('.')):$('span',{class:'chip'},n))))];
  add($('h2',{},k.name),kv({kind:k.kind,implementation:k.implementation,step:k.step,block:k.block,'loop depth':k.depth||null}),
   part('reads',k.reads.filter(n=>!k.writes.includes(n))),part('writes',k.writes.filter(n=>!k.reads.includes(n))),part('reads and writes',k.reads.filter(n=>k.writes.includes(n))))}
 if(type==='field'){const f=declFields[id]||{},p=planFields[id];const ks=touching(id);
  add($('h2',{},id),$('div',{class:'muted'},f.description||(p&&p.description)),
   kv({category:f.category,dtype:f.dtype,shape:'('+(f.shape||[]).join(', ')+')',default:f.computed?'computed':f.default,units:f.units,bounds:f.bounds,
    expression:f.expression,'active now':f.active==null?'module closed':f.active?'yes':'no',
    resident:p&&p.runtime?`${p.runtime.dtype}[${p.runtime.shape.join('×')}] ${fmtBytes(p.runtime.bytes)} on ${p.runtime.device}`:null,
    coordinate:p&&(p.dim_coords||(p.coordinate?'(coordinate)':null)),references:p&&p.references,'partition by':p&&p.partition_by,output:f.output}),
   (f.depends_on||[]).length?[$('h3',{},'active when'),f.depends_on.map(cond)]:null,(f.required_by||[]).length?[$('h3',{},'required by'),f.required_by.map(cond)]:null,
   (f.materialize||[]).length?[$('h3',{},'materialized when'),f.materialize.map(cond)]:null,
   $('h3',{},'kernels ('+ks.length+')'),ks.map(k=>$('div',{},$('span',{class:k.writes.includes(id)?(k.reads.includes(id)?'RW':'W'):'R'},k.writes.includes(id)?(k.reads.includes(id)?'RW ':'W '):'R '),link('kernel',k.id,k.name))),
   (gating=>gating.length?[$('h3',{},'options gating it'),gating.map(o=>link('option',o.path,o.path))]:null)(X.options.filter(o=>(o.used_by||[]).some(u=>u.split(' ')[0]===id))))}
 if(type==='option'){const o=options[id];
  add($('h2',{},'option ',id),$('div',{class:'muted'},o.description),kv({type:o.type,default:o.default,value:o.value,changed:o.changed?'yes':null,constraints:o.constraints,'set with':o.set}),
   (o.relevant_when||[]).filter(Boolean).length?[$('h3',{},'relevant when'),o.relevant_when.filter(Boolean).map(cond)]:null,
   o.choices?[$('h3',{},'choices'),table([['choice',c=>$(c.selected?'b':'span',{},c.choice)],['code',c=>c.code],['needs',c=>c.requires.map(m=>link('module',m,m))],['conflicts',c=>c.conflicts.map(conflict)]],o.choices)]:null,
   (o.conflicts||[]).length?[$('h3',{},'conflicts'),o.conflicts.map(conflict)]:null,
   $('h3',{},'gates ('+(o.used_by||[]).length+')'),(o.used_by||[]).map(u=>declFields[u.split(' ')[0]]?link('field',u.split(' ')[0],u):$('span',{class:'chip'},u)))}}

// ---------------------------------------------------------------- module map
function layered(nodes,edges){ // nodes {id,depth,rank}; returns order per column (barycentre sweeps)
 const cols={};for(const n of nodes)(cols[n.depth]=cols[n.depth]||[]).push(n);
 const keys=Object.keys(cols).map(Number).sort((a,b)=>a-b);for(const k of keys)cols[k].sort((a,b)=>a.rank-b.rank);
 const pos={};const place=()=>{for(const k of keys)cols[k].forEach((n,i)=>pos[n.id]=i)};place();
 const nb={};for(const[a,b]of edges){(nb[a]=nb[a]||[]).push(b);(nb[b]=nb[b]||[]).push(a)}
 for(let sweep=0;sweep<4;sweep++){for(const k of(sweep%2?keys.slice().reverse():keys)){const bc=n=>{const ps=(nb[n.id]||[]).map(m=>pos[m]).filter(x=>x!=null);return ps.length?ps.reduce((a,b)=>a+b,0)/ps.length:pos[n.id]};
  cols[k].sort((a,b)=>bc(a)-bc(b)||a.rank-b.rank);place()}}
 return{cols,keys}}
function moduleMap(){
 const shown=X.modules.filter(m=>full()||m.opened);const ids=new Set(shown.map(m=>m.name));
 const nodes=shown.map(m=>({id:m.name,kind:'module',depth:(planMods[m.name]||{}).depth||0,rank:order.includes(m.name)?order.indexOf(m.name):100+X.modules.indexOf(m)}));
 const edges=[];for(const m of shown)for(const r of m.references){const name=r.replace(/\?$/,'');if(ids.has(name))edges.push([m.name,name,r.endsWith('?')?'optional':'','ref'])}
 if(full())for(const m of shown)for(const c of m.conflicts)if(ids.has(c)&&m.name<c)edges.push([m.name,c,'','conflict']);
 // Options whose choices open modules (every choice in the full declaration,
 // the selected one now) and options gating fields, by module.
 const opts=[];const optNode=o=>{if(!opts.some(n=>n.id==='opt:'+o.path))opts.push({id:'opt:'+o.path,path:o.path,kind:'option',depth:-1,rank:opts.length,h:30})};
 for(const o of X.options)for(const c of o.choices||[])if(c.requires.length&&(full()||c.selected))for(const m of c.requires)if(ids.has(m)){optNode(o);edges.push(['opt:'+o.path,m,c.choice,'opt'])}
 for(const o of X.options){const per={};for(const u of o.used_by||[]){const f=declFields[u.split(' ')[0]];if(f&&(full()||f.active)&&ids.has(f.module))per[f.module]=(per[f.module]||0)+1}
  for(const[m,n]of Object.entries(per)){optNode(o);edges.push(['opt:'+o.path,m,n+(full()?' fields':' active'),'gate'])}}
 const all=[...opts,...nodes];const{cols,keys}=layered(all,edges.filter(e=>e[3]!=='conflict'));
 const W=210,H=48,GX=90,GY=12,PAD=16;const at={},hs={};let maxY=0;keys.forEach((k,ci)=>{let y=PAD;cols[k].forEach(n=>{hs[n.id]=n.h||H;at[n.id]=[PAD+ci*(W+GX),y];y+=hs[n.id]+GY;maxY=Math.max(maxY,y)})});
 const svg=mk('svg',{width:PAD*2+keys.length*(W+GX)-GX,height:maxY+PAD*2});
 const defs=mk('defs');for(const[id,color]of[['a','var(--muted)'],['o','var(--opt)'],['h','var(--accent)']]){const m=mk('marker',{id:'mk'+id,viewBox:'0 0 10 10',refX:10,refY:5,markerWidth:7,markerHeight:7,orient:'auto'});
  m.append(mk('path',{d:'M0,0L10,5L0,10z',style:`fill:${color}`}));defs.append(m)}svg.append(defs);
 const sel=S.sel&&S.sel.type==='module'?S.sel.id:S.sel&&S.sel.type==='option'?'opt:'+S.sel.id:null;
 const near=new Set(sel?[sel,...edges.filter(e=>e[0]===sel||e[1]===sel).flatMap(e=>[e[0],e[1]])]:[]);
 for(const[a,b,label,kind]of edges){if(!at[a]||!at[b])continue;const[ax,ay]=at[a],[bx,by]=at[b];let d;
  if(kind==='conflict')d=`M${ax+W/2},${ay+H} C${ax+W/2},${ay+H+30} ${bx+W/2},${by+H+30} ${bx+W/2},${by+H}`;
  else{const right=bx>ax;const sx=right?ax+W:ax,tx=right?bx:bx+W,sy=ay+hs[a]/2,ty=by+hs[b]/2,c=Math.max(30,Math.abs(tx-sx)/2);d=`M${sx},${sy} C${sx+(right?c:-c)},${sy} ${tx+(right?-c:c)},${ty} ${tx},${ty}`}
  const hot=sel&&(a===sel||b===sel);const optional=kind==='opt'||kind==='gate';const p=mk('path',{d,class:'edge '+(optional?'opt ':kind==='conflict'?'conflict ':'')+(hot?'hot':sel?'dim':''),
   'marker-end':kind==='conflict'?null:`url(#mk${hot?'h':optional?'o':'a'})`});
  p.append(mk('title',{},kind==='conflict'?`${a} conflicts with ${b}`:kind==='opt'?`${a.slice(4)}=${label} needs ${b}`:kind==='gate'?`${a.slice(4)} gates ${label} of ${b}`:`${a} uses ${b}${label?' ('+label+')':''}`));svg.append(p);
  if(label&&optional&&(hot||!sel)){svg.append(mk('text',{x:ax+W+8,y:ay+hs[a]/2-4,class:'muted',style:'font-size:10px'},fit(label,14)))}}
 for(const n of all){const[x,y]=at[n.id];const m=declMods[n.id];const opened=n.kind==='option'||m.opened;
  const active=n.kind==='module'?X.fields.filter(f=>f.module===n.id&&f.active).length:0;
  const g=mk('g',{class:'node '+(n.kind==='option'?'option ':'')+(opened?'':'closed ')+((sel===n.id)?'sel ':'')+(sel&&!near.has(n.id)?'dim ':'')+(S.q&&!match(n.id)?'dim':''),transform:`translate(${x},${y})`});
  g.append(mk('rect',{width:W,height:hs[n.id],rx:8}));
  if(n.kind==='option'){const o=options[n.path];g.append(mk('text',{x:10,y:19,style:o.changed?'font-weight:700':''},fit(n.path+' = '+val(o.value),30)));
   g.addEventListener('click',()=>select('option',n.path))}
  else{const p=planMods[n.id]||{};const tags=[m.required?null:'optional',opened?null:'closed',p.requirement&&p.requirement.halo?'halo':null].filter(Boolean);
   g.append(mk('text',{x:10,y:19,style:'font-weight:600'},n.id),mk('text',{x:10,y:36,class:'muted',style:'font-size:11px'},
    fit([active+(full()?'/'+m.fields:'')+' fields',moduleKernels(n.id).length?moduleKernels(n.id).length+' kernels':null,...tags].filter(Boolean).join(' · '),32)));
   g.addEventListener('click',()=>select('module',n.id))}
  g.append(mk('title',{},n.kind==='option'?n.path:(m.class+'\n'+(m.description||''))));svg.append(g)}
 return $('div',{style:'overflow:auto'},svg)}

// ---------------------------------------------------------------- execution swimlanes
function swimlanes(){
 const used=new Set(kernels.flatMap(k=>[...k.reads,...k.writes].map(laneOf)));
 const lanes=[...order.filter(m=>used.has(m)),...[...used].filter(l=>!order.includes(l)).sort()];
 const LW=300,CW=76,RH=24,HH=96,PAD=10;const width=LW+lanes.length*CW+PAD,height=HH+K.length*RH+PAD;
 const svg=mk('svg',{width,height});
 const selModule=S.sel&&S.sel.type==='module'?S.sel.id:null,selField=S.sel&&S.sel.type==='field'?S.sel.id:null,selKernel=S.sel&&S.sel.type==='kernel'?S.sel.id:null;
 lanes.forEach((l,i)=>{const x=LW+i*CW;svg.append(mk('rect',{x,y:HH-6,width:CW,height:K.length*RH+6,style:`fill:${l===selModule?'var(--soft)':i%2?'var(--lane)':'transparent'}`}));
  const h=mk('g',{class:'lanehead',transform:`translate(${x+CW/2},${HH-12}) rotate(-35)`});h.append(mk('text',{x:0,y:0,style:`font-weight:${l===selModule?700:500}`},fit(l,18)),mk('title',{},l));
  if(declMods[l])h.addEventListener('click',()=>select('module',l));svg.append(h)});
 // Loop brackets in the gutter.
 const spans={};K.forEach((k,i)=>{for(const id of k.loops||[]){const s=spans[id]=spans[id]||[i,i];s[1]=i}});
 K.forEach((k,i)=>{const y=HH+i*RH;
  if(k.head){svg.append(mk('rect',{x:0,y,width,height:RH,style:'fill:var(--chip)'}),mk('text',{x:8,y:y+16,style:'font-weight:700'},k.label==='eager launches'?'eager launches (initialization and step bodies, first-call order)':k.step+' · '+k.label));return}
  const lane=new Set([...k.reads,...k.writes].map(laneOf));
  const dim=(selModule&&!lane.has(selModule))||(selField&&!k.reads.includes(selField)&&!k.writes.includes(selField))||(S.q&&!match(k.name)&&![...k.reads,...k.writes].some(match));
  const g=mk('g',{class:'row'+(selKernel===k.id?' sel':'')+(dim?' dim':'')});g.append(mk('rect',{class:'bg',x:0,y,width,height:RH}));
  const indent=8+k.depth*16;const color={kernel:'var(--accent)',torch:'var(--muted)',collective:'var(--warn)',loop:'var(--rw)'}[k.kind]||'var(--muted)';
  g.append(mk('circle',{cx:indent+4,cy:y+RH/2,r:k.kind==='loop'?5:4,style:`fill:${k.kind==='loop'?'none':color};stroke:${color};stroke-width:1.5`}));
  g.append(mk('text',{x:indent+14,y:y+16,style:k.kind==='loop'?'font-style:italic':''},fit((k.kind==='loop'?'⟳ ':'')+k.name,Math.floor((LW-indent-70)/7))));
  g.append(mk('text',{x:LW-8,y:y+16,'text-anchor':'end',class:'muted',style:'font-size:10px'},k.implementation||k.kind));
  lanes.forEach((l,j)=>{const r=k.reads.filter(n=>laneOf(n)===l),w=k.writes.filter(n=>laneOf(n)===l);if(!r.length&&!w.length)return;
   const kind=r.length&&w.length?'RW':w.length?'W':'R';const x=LW+j*CW+8,count=new Set([...r,...w]).size;const hit=selField&&(r.includes(selField)||w.includes(selField));
   g.append(mk('rect',{x,y:y+4,width:CW-16,height:RH-8,rx:8,style:`fill:var(--${kind.toLowerCase()});opacity:${hit?1:.85};${hit?'stroke:var(--fg);stroke-width:2':''}`}));
   g.append(mk('text',{x:x+(CW-16)/2,y:y+16,'text-anchor':'middle',style:'fill:#fff;font-size:11px;font-weight:600'},kind+' '+count));
   g.append(mk('title',{},`${k.name} · ${l}\nreads: ${r.map(n=>n.split('.').pop()).join(', ')||'—'}\nwrites: ${w.map(n=>n.split('.').pop()).join(', ')||'—'}`))});
  g.addEventListener('click',()=>select('kernel',k.id));svg.append(g)});
 for(const[id,[a,b]]of Object.entries(spans)){const owner=K.find(k=>k.loopId==Number(id));const x=4+(owner?owner.depth:0)*16;
  svg.append(mk('path',{d:`M${x+6},${HH+a*RH+4} L${x},${HH+a*RH+4} L${x},${HH+(b+1)*RH-4} L${x+6},${HH+(b+1)*RH-4}`,style:'fill:none;stroke:var(--rw);stroke-width:1.5'}))}
 return $('div',{style:'overflow:auto'},svg)}

// ---------------------------------------------------------------- sections
const sections=[['map','Map'],['options','Options'],['fields','Fields'],['partition','Coordinates & partition'],['outputs','Outputs'],['overview','Overview']];
function mapView(){const div=$('div');
 div.append(card(full()?'Modules — everything declared':'Modules — opened',moduleMap(),$('div',{class:'legend'},
  $('span',{},'grey arrow: uses (references)'),full()?$('span',{},'dashed box: closed module · red dotted: conflicts'):null,
  $('span',{style:'color:var(--opt)'},full()?'orange: options, with the module a choice needs or the fields they gate':'orange: options gating active fields (bold: changed)'),
  $('span',{},'click a node for details'))));
 div.append(card('Execution — kernels in launch order across module lanes',kernels.length?swimlanes():$('div',{class:'muted'},
  'No kernels recorded yet: export after the model has run its managed steps to see every kernel and the fields it touches.'),
  kernels.length?$('div',{class:'legend'},$('span',{},$('i',{style:'background:var(--r)'}),'reads'),$('span',{},$('i',{style:'background:var(--w)'}),'writes'),
   $('span',{},$('i',{style:'background:var(--rw)'}),'reads and writes'),$('span',{},'number: fields of that module'),$('span',{},'purple bracket: loop body'),$('span',{},'click a row or lane')):null));
 return div}
function optionsView(){const div=$('div'),body=$('div'),only=$('input',{type:'checkbox'});const depth=p=>p.split('.').length-1;
 const draw=()=>{const rows=X.options.filter(o=>(!only.checked||o.changed)&&(!S.q||JSON.stringify(o).toLowerCase().includes(S.q)));body.innerHTML='';
  const name=o=>$('span',{style:`padding-left:${depth(o.path)*14}px`},$(o.kind==='record'?'i':'b',{},o.path.split('.').pop()),depth(o.path)?$('div',{class:'muted mono'},o.path):null);
  const value=o=>o.kind==='record'?'':$(o.changed?'b':'span',{class:'mono'},val(o.value));
  const relevant=o=>(Array.isArray(o.relevant_when)?o.relevant_when:[o.relevant_when]).filter(Boolean).map(cond);
  body.append(full()?table([['option',name],['kind',o=>o.kind],['type',o=>$('span',{class:'mono'},o.type)],['default',o=>$('span',{class:'mono'},val(o.default))],['value',value],
   ['choices',o=>o.choices?$('table',{},o.choices.map(c=>$('tr',{},$('td',{},$(c.selected?'b':'span',{},c.choice)),$('td',{class:'muted'},'code '+c.code),
    $('td',{},c.requires.map(m=>link('module',m,'needs '+m))),$('td',{},c.backends?chips(c.backends):''),$('td',{},Object.keys(c.traits||{}).length?$('span',{class:'mono'},JSON.stringify(c.traits)):''),
    $('td',{},c.conflicts.map(conflict))))):(o.conflicts||[]).map(conflict)],['constraints',o=>chips(o.constraints)],['relevant when',relevant],
   ['parameters',o=>o.affects_parameters===true?'yes':o.affects_parameters?cond(o.affects_parameters):''],['gates',o=>(o.used_by||[]).length?(o.used_by.length+' items'):''],
   ['set with',o=>o.set?$('span',{class:'mono'},o.set):''],['description',o=>o.description]],rows,{rowClass:o=>o.kind==='record'?'off':'',onrow:o=>o.kind!=='record'&&select('option',o.path)})
  :table([['option',name],['value',value],['default',o=>o.kind==='record'?'':$('span',{class:'mono'},val(o.default))],['relevant',relevant],['description',o=>o.description]],
   rows,{rowClass:o=>o.kind==='record'?'off':'',onrow:o=>o.kind!=='record'&&select('option',o.path)}))};
 only.addEventListener('input',draw);
 div.append(card('Options · '+X.options_class,$('div',{class:'muted'},full()?'Every option with its type, default, choices (codes, needed modules, backends, traits, conflicts), relevance and what it gates. ✓ holds now · ✗ does not · ⚠ would conflict with the current settings.':
  'The current option values (bold: changed from the default). Switch to “Full declaration” for every choice, conflict and gate.'),$('div',{class:'controls'},$('label',{},only,' changed only')),body));draw();return div}
function fieldsView(){const div=$('div'),body=$('div');const mods=full()?X.modules.map(m=>m.name):D.overview.opened_modules;
 const mod=$('select',{},$('option',{value:''},'all modules'),mods.map(m=>$('option',{value:m},m)));
 const rowsAll=full()?X.fields:D.fields.filter(f=>f.active);const cats=[...new Set(rowsAll.map(f=>f.category))].sort();
 const cat=$('select',{},$('option',{value:''},'all categories'),cats.map(c=>$('option',{value:c},c)));
 const draw=()=>{const rows=rowsAll.filter(f=>(!mod.value||f.module===mod.value)&&(!cat.value||f.category===cat.value)&&(!S.q||JSON.stringify(f).toLowerCase().includes(S.q)));body.innerHTML='';
  const kernelsOf=f=>{const n=f.module+'.'+f.name,ks=touching(n);return ks.length?ks.length:''};
  body.append($('div',{class:'muted'},rows.length+(full()?' declared':' active')+' fields'),full()?table([['module',f=>f.module],['field',f=>$('b',{},f.name)],['category',f=>f.category],['dtype',f=>f.dtype],
   ['shape',f=>$('span',{class:'mono'},'('+f.shape.join(', ')+')')],['default',f=>f.computed?$('span',{class:'muted'},'computed'):$('span',{class:'mono'},val(f.default))],['bounds',f=>chips(f.bounds)],
   ['active when',f=>f.depends_on.map(cond)],['required by',f=>f.required_by.map(cond)],['active now',f=>f.active==null?$('span',{class:'muted'},'module closed'):f.active?'yes':'no'],
   ['units',f=>f.units],['description',f=>f.description]],rows,{rowClass:f=>f.active?'':'off',onrow:f=>select('field',f.module+'.'+f.name)})
  :table([['module',f=>f.module],['field',f=>$('b',{},f.name)],['category',f=>f.category],['dtype',f=>f.dtype],['shape',f=>$('span',{class:'mono'},'('+f.shape.join(', ')+')')],
   ['resident',f=>f.runtime?$('span',{class:'mono'},`${f.runtime.dtype}[${f.runtime.shape.join('×')}] ${fmtBytes(f.runtime.bytes)}`):''],['coordinate',f=>f.dim_coords||(f.coordinate?'(coordinate)':'')],
   ['links',f=>chips([f.partition_by&&'by '+f.partition_by,f.references&&'→ '+f.references,f.selects&&'selects '+f.selects].filter(Boolean))],['kernels',kernelsOf],
   ['flags',f=>chips([f.computed&&'computed',f.reference_index&&'index',f.after_init&&'after_init',f.key&&'key',f.mode!=='device'&&f.mode].filter(Boolean))],
   ['units',f=>f.units],['description',f=>f.description]],rows,{onrow:f=>select('field',f.module+'.'+f.name)}))};
 for(const e of[mod,cat])e.addEventListener('input',draw);const total=D.fields.reduce((a,f)=>a+(f.runtime?f.runtime.bytes:0),0);
 div.append(card(null,$('div',{class:'controls'},mod,cat,total?$('span',{class:'muted'},'resident tensors: '+fmtBytes(total)):null),body));draw();return div}
function graph(nodes,edges){const W=220,H=44,GX=60,GY=14,PAD=16;const{cols,keys}=layered(nodes,edges);const at={};let maxY=0;
 keys.forEach((k,ci)=>cols[k].forEach((n,i)=>{at[n.id]=[PAD+ci*(W+GX),PAD+i*(H+GY)];maxY=Math.max(maxY,at[n.id][1])}));
 const svg=mk('svg',{width:PAD*2+keys.length*(W+GX)-GX,height:maxY+H+PAD*2});const defs=mk('defs'),m=mk('marker',{id:'mkc',viewBox:'0 0 10 10',refX:10,refY:5,markerWidth:7,markerHeight:7,orient:'auto'});
 m.append(mk('path',{d:'M0,0L10,5L0,10z',style:'fill:var(--muted)'}));defs.append(m);svg.append(defs);
 for(const[a,b]of edges){if(!at[a]||!at[b])continue;const[ax,ay]=at[a],[bx,by]=at[b];const sx=ax,tx=bx+W,sy=ay+H/2,ty=by+H/2;
  svg.append(mk('path',{class:'edge',d:`M${sx},${sy} C${sx-GX/2},${sy} ${tx+GX/2},${ty} ${tx},${ty}`,'marker-end':'url(#mkc)'}))}
 for(const n of nodes){const[x,y]=at[n.id];const g=mk('g',{class:'node '+(n.cls||''),transform:`translate(${x},${y})`});g.append(mk('rect',{width:W,height:H,rx:8}),
  mk('text',{x:10,y:18,style:'font-weight:600'},fit(n.id,30)),mk('text',{x:10,y:34,class:'muted',style:'font-size:11px'},fit(n.sub,34)),mk('title',{},n.title||n.id));svg.append(g)}
 return $('div',{style:'overflow:auto'},svg)}
function partitionView(){const div=$('div'),P=D.partition;
 const nodes=D.coordinates.map(c=>({id:c.name,depth:c.depth,rank:0,cls:c.name===P.key?'key':'',sub:[c.module,c.extent!=null?c.extent+' rows':null,c.via&&c.via!==c.name?'by '+c.via:null].filter(Boolean).join(' · '),
  title:[c.name,c.module&&'module '+c.module,c.extent!=null&&c.extent+' rows',c.via&&'owned through '+c.via,c.selection&&'outputs select it by '+c.selection].filter(Boolean).join('\n')}));
 div.append(card('Coordinate ownership (arrow: rows live with)',graph(nodes,D.coordinates.filter(c=>c.target).map(c=>[c.name,c.target])),$('div',{class:'legend'},'highlighted: the partition key; every partitioned coordinate follows it')));
 div.append(card('Partition',kv({key:P.key,group:P.group,rank:P.spatial_world_size>1?P.spatial_rank+' of '+P.spatial_world_size:'single rank','imbalance tolerance':P.imbalance_tolerance}),
  P.balance?[$('h3',{},'balance'),kv({'groups cut':P.balance.split?'yes':'no',pieces:P.balance.pieces,'imbalance':(100*P.balance.imbalance).toFixed(2)+' %','edges between ranks':P.balance.cut})]:null,
  P.halo?[$('h3',{},'halo'),kv({transport:P.halo.transport,skipped:P.halo.skipped,owned:P.halo.owned,ghosts:P.halo.ghosts,'ghosts (all ranks)':P.halo.ghosts_total,sends:P.halo.sends,receives:P.halo.receives})]
   :$('div',{class:'muted'},'No halo: every partition group is closed.'),
  Object.keys(P.reduce||{}).length?[$('h3',{},'all-reduce transports'),Object.entries(P.reduce).map(([g,r])=>kv({group:g,transports:r.transports,skipped:r.skipped}))]:null));
 const refs=D.fields.filter(f=>f.references&&!f.coordinate);
 if(refs.length)div.append(card('Reference fields',table([['field',f=>link('field',f.module+'.'+f.name,f.module+'.'+f.name)],['on',f=>f.dim_coords],['references',f=>f.references],
  ['ownership',f=>D.coordinates.some(c=>c.via===f.name)?'defines partition':'may cross groups']],refs)));return div}
function outputsView(){const o=D.outputs,div=$('div');div.append(card(null,kv({directory:o.directory,sink:o.sink,windows:o.windows})));
 if(Object.keys(o.views).length)div.append(card('Views',kv(o.views)));const byName={};for(const v of o.variables)(byName[v.name]=byName[v.name]||{...v,operations:[]}).operations.push(v.operation);
 div.append(card('Variables',o.variables.length?table([['variable',v=>$('b',{},v.name)],['operations',v=>chips(v.operations)],['expression',v=>$('span',{class:'mono'},v.expression)],
  ['view',v=>v.view],['axes',v=>chips(v.axes)],['shape',v=>v.shape?'('+v.shape.join(', ')+')':'']],Object.values(byName)):$('div',{class:'muted'},'No statistics outputs declared.')));return div}
function overviewView(){const div=$('div');div.append(card('Model',kv(D.overview)));
 div.append(card('Model contract',kv({partition:X.partition,'grid axes':X.grid_axes,'kernel fields':X.kernel_fields,'step field providers':X.step_fields.providers,'step field expressions':X.step_fields.expressions}),
  $('h3',{},'backends'),Object.keys(X.backends).length?kv(X.backends):$('div',{class:'muted'},'No backend-specific requirements.'),
  $('h3',{},'managed steps'),X.steps.map(t=>$('details',{open:''},$('summary',{},$('b',{},t.name)),$('pre',{class:'muted'},t.doc))),
  X.initialize?[$('h3',{},'initialize_model_state'),$('pre',{class:'muted'},X.initialize)]:null));
 if(Object.keys(D.dimensions).length)div.append(card('Dimensions',kv(D.dimensions)));if(Object.keys(D.conditions).length)div.append(card('Conditions',kv(D.conditions)));return div}
const views={map:mapView,options:optionsView,fields:fieldsView,partition:partitionView,outputs:outputsView,overview:overviewView};

// ---------------------------------------------------------------- shell
document.getElementById('title').textContent=D.title;
document.getElementById('subtitle').textContent=[D.overview.backend,D.overview.device,D.overview.state,D.overview.opened_modules.length+' of '+X.modules.length+' modules open'].join(' · ');
const nav=document.getElementById('nav');for(const[id,label]of sections)nav.append($('button',{'data-tab':id,onclick:()=>{S.tab=id;render()}},label));
for(const b of document.querySelectorAll('#mode button'))b.addEventListener('click',()=>{S.mode=b.dataset.mode;render()});
document.getElementById('q').addEventListener('input',e=>{S.q=e.target.value.toLowerCase();render()});
function render(){for(const b of nav.children)b.classList.toggle('on',b.dataset.tab===S.tab);
 if(S.sel&&S.sel.type==='kernel'&&!K[S.sel.id])S.sel=null;
 for(const b of document.querySelectorAll('#mode button'))b.classList.toggle('on',b.dataset.mode===S.mode);
 const main=document.getElementById('main');main.innerHTML='';main.append(views[S.tab]());drawer();
 history.replaceState(null,'',`#${S.tab}${S.mode==='full'?'+full':''}${S.sel?'@'+S.sel.type+':'+S.sel.id:''}`)}
// #tab[+full][@type:id] restores the view, the mode and a selection.
{const[view,sel]=decodeURIComponent(location.hash.slice(1)||'map').split('@');const[t,m]=view.split('+');if(views[t])S.tab=t;if(m==='full')S.mode='full';
 if(sel){const i=sel.indexOf(':'),type=sel.slice(0,i),id=sel.slice(i+1);S.sel={type,id:type==='kernel'?Number(id):id}}}
render();
</script></body></html>
"""

__all__ = ["model_structure", "render_html"]

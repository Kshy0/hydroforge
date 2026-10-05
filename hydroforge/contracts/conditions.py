"""Cold, strict resolution of reserved ``options.`` field conditions.

Only declarations are traversed. This module never evaluates properties, model
runtime state, expressions or tensor truthiness.
"""

from __future__ import annotations

import inspect
from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import Any, get_args

from hydroforge.contracts.options import OptionsConfig


def is_option_condition(value: str) -> bool:
    return value.startswith("options.")


def validate_condition(value: str) -> str:
    if value.isidentifier():
        return value
    parts = value.split(".")
    if (
        len(parts) < 2
        or parts[0] != "options"
        or any(not part.isidentifier() or part.startswith("_") for part in parts[1:])
    ):
        raise ValueError(
            "depends_on requires a module identifier or options.<public bool path>"
        )
    return value


def module_conditions(values: Iterable[str]) -> tuple[str, ...]:
    return tuple(value for value in values if not is_option_condition(value))


def resolve_option_condition(options: OptionsConfig, path: str) -> bool:
    """Read an exact declared bool through frozen, declared options records."""
    validate_condition(path)
    if not is_option_condition(path):
        raise ValueError(f"not an options condition: {path!r}")
    current = options
    parts = path.split(".")[1:]
    for index, part in enumerate(parts):
        if not isinstance(current, OptionsConfig) or not type(current).model_config.get(
            "frozen"
        ):
            raise ValueError(
                f"condition {path!r} requires frozen OptionsConfig records"
            )
        field = type(current).model_fields.get(part)
        if field is None:
            raise ValueError(
                f"condition {path!r} names an unknown declared option {part!r}"
            )
        descriptor = inspect.getattr_static(type(current), part, None)
        if descriptor is not None and hasattr(descriptor, "__get__"):
            raise ValueError(f"condition {path!r} cannot traverse descriptors")
        # Read stored declared fields directly, never __getattr__/properties.
        values = object.__getattribute__(current, "__dict__")
        if part not in values:
            raise ValueError(f"condition {path!r} has no stored value for {part!r}")
        value = values[part]
        if index == len(parts) - 1:
            if field.annotation is not bool or type(value) is not bool:
                raise ValueError(f"condition {path!r} requires a declared exact bool")
            if part in current._coerced_bool_fields:
                raise ValueError(
                    f"condition {path!r} requires exact bool input, not a coerced value"
                )
            return value
        annotations = get_args(field.annotation) or (field.annotation,)
        records = tuple(item for item in annotations if item is not type(None))
        if not records or any(
            not isinstance(item, type) or not issubclass(item, OptionsConfig)
            for item in records
        ):
            raise ValueError(
                f"condition {path!r} requires a declared OptionsConfig path"
            )
        current = value
    raise ValueError(f"empty condition {path!r}")


def resolve_conditions(
    modules: Iterable[Any], options: OptionsConfig | None
) -> Mapping[str, bool]:
    """Resolve every referenced gate of opened owners, including false ANDs."""
    values: dict[str, bool] = {}
    for module in modules:
        for field in module.tensor_fields.values():
            for condition in field.tensor.depends_on:
                if not is_option_condition(condition) or condition in values:
                    continue
                try:
                    if options is None:
                        raise ValueError("provide an explicit options context")
                    values[condition] = resolve_option_condition(options, condition)
                except ValueError as error:
                    raise ValueError(f"{module.name}.{field.name}: {error}") from error
    return MappingProxyType(values)


def conditions_satisfied(
    dependencies: Iterable[str],
    opened_modules: Iterable[str],
    conditions: Mapping[str, bool] | None = None,
) -> bool:
    opened = set(opened_modules)
    results = []
    for dependency in dependencies:
        if is_option_condition(dependency):
            if conditions is None or dependency not in conditions:
                raise ValueError(
                    f"unresolved options condition {dependency!r}; provide an options context"
                )
            value = conditions[dependency]
            if type(value) is not bool:
                raise TypeError(
                    f"resolved condition {dependency!r} must be an exact bool"
                )
            results.append(value)
        else:
            results.append(dependency in opened)
    return all(results)

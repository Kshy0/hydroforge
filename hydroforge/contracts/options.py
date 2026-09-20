"""Typed immutable model-option declarations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from copy import deepcopy
from enum import Enum
from types import MappingProxyType
from typing import Any, Self

from pydantic import ConfigDict, Field, PrivateAttr, create_model, model_validator
from pydantic.fields import FieldInfo

from hydroforge.contracts.validation import HydroForgeModel

_OPTIONS_METADATA = "hydroforge_options"


def _choice_token(value: Any, *, label: str) -> str:
    raw = value.value if isinstance(value, Enum) else value
    if type(raw) is not str or not raw:
        raise TypeError(f"{label} must resolve to a non-empty string token")
    return raw


def _choice_mapping(
    values: Mapping[Any, Any],
    *,
    label: str,
    value_type: type,
) -> dict[str, Any]:
    if not isinstance(values, Mapping) or not values:
        raise TypeError(f"{label} must be a non-empty mapping")
    result: dict[str, Any] = {}
    for choice, value in values.items():
        token = _choice_token(choice, label=f"{label} choice")
        if token in result:
            raise ValueError(f"{label} repeats choice token {token!r}")
        if value_type is int:
            if type(value) is not int:
                raise TypeError(f"{label}[{token!r}] must be an exact int")
        elif value_type is tuple:
            if type(value) is not tuple or any(
                type(item) is not str or not item.isidentifier() for item in value
            ):
                raise TypeError(f"{label}[{token!r}] must be a tuple of identifiers")
        result[token] = value
    return result


def _backend_support(
    values: Mapping[str, tuple[Any, ...]] | None,
    *,
    choices: Mapping[str, Any],
) -> dict[str, tuple[str, ...]]:
    result: dict[str, tuple[str, ...]] = {}
    for backend, supported in (values or {}).items():
        if type(backend) is not str or not backend.isidentifier():
            raise TypeError("option backend names must be identifiers")
        if type(supported) is not tuple or not supported:
            raise TypeError(
                f"supported_backends[{backend!r}] must be a non-empty tuple"
            )
        tokens = tuple(
            _choice_token(choice, label=f"backend {backend!r} option")
            for choice in supported
        )
        unknown = set(tokens).difference(choices)
        if unknown:
            raise ValueError(
                f"backend {backend!r} contains unknown option choices: "
                f"{sorted(unknown)}"
            )
        if len(tokens) != len(set(tokens)):
            raise ValueError(f"backend {backend!r} repeats an option choice")
        result[backend] = tokens
    return result


def OptionField(
    default: Any,
    *,
    codes: Mapping[Any, int],
    requires_modules: Mapping[Any, tuple[str, ...]] | None = None,
    supported_backends: Mapping[str, tuple[Any, ...]] | None = None,
    description: str,
) -> FieldInfo:
    """Declare one user-visible option with a stable device code."""

    normalized_codes = _choice_mapping(codes, label="option codes", value_type=int)
    if len(set(normalized_codes.values())) != len(normalized_codes):
        raise ValueError("option codes must be unique")
    default_token = _choice_token(default, label="option default")
    if default_token not in normalized_codes:
        raise ValueError(f"option default {default_token!r} is absent from codes")
    requirements = _choice_mapping(
        requires_modules or {choice: () for choice in normalized_codes},
        label="option module requirements",
        value_type=tuple,
    )
    unknown = set(requirements).difference(normalized_codes)
    if unknown:
        raise ValueError(
            f"option module requirements contain unknown choices: {sorted(unknown)}"
        )
    complete_requirements = {
        choice: tuple(requirements.get(choice, ())) for choice in normalized_codes
    }
    return Field(
        default=default,
        description=description,
        json_schema_extra={
            _OPTIONS_METADATA: {
                "role": "option",
                "codes": normalized_codes,
                "requires_modules": complete_requirements,
                "supported_backends": _backend_support(
                    supported_backends,
                    choices=normalized_codes,
                ),
            }
        },
    )


def ForcingOptionField(
    default: Any,
    *,
    codes: Mapping[Any, int],
    requires: Mapping[Any, tuple[str, ...]],
    supported_backends: Mapping[str, tuple[Any, ...]] | None = None,
    description: str,
) -> FieldInfo:
    """Declare one forcing-input option and its exact required arguments."""

    normalized = _choice_mapping(
        requires,
        label="forcing requirements",
        value_type=tuple,
    )
    normalized_codes = _choice_mapping(
        codes,
        label="forcing option codes",
        value_type=int,
    )
    if len(set(normalized_codes.values())) != len(normalized_codes):
        raise ValueError("forcing option codes must be unique")
    if set(normalized_codes) != set(normalized):
        raise ValueError("forcing option codes and requirements differ")
    default_token = _choice_token(default, label="forcing option default")
    if default_token not in normalized:
        raise ValueError(
            f"forcing option default {default_token!r} is absent from requirements"
        )
    return Field(
        default=default,
        description=description,
        json_schema_extra={
            _OPTIONS_METADATA: {
                "role": "forcing",
                "codes": normalized_codes,
                "requires": normalized,
                "supported_backends": _backend_support(
                    supported_backends,
                    choices=normalized_codes,
                ),
            }
        },
    )


class OptionsConfig(HydroForgeModel):
    """Immutable nested model options, separate from tensor input."""

    model_config = ConfigDict(strict=False)
    _forcing_rules: (
        tuple[tuple[str, str, frozenset[str], frozenset[str]], ...] | None
    ) = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_declared_choices(self) -> Self:
        for name, field in type(self).model_fields.items():
            metadata = self._field_metadata(field)
            if metadata is None:
                continue
            token = _choice_token(getattr(self, name), label=f"option field {name}")
            choices = metadata["codes"]
            if token not in choices:
                raise ValueError(
                    f"option field {name!r} has unsupported choice {token!r}; "
                    f"expected one of {sorted(choices)}"
                )
        return self

    @staticmethod
    def _field_metadata(field: FieldInfo) -> Mapping[str, Any] | None:
        extra = field.json_schema_extra
        if not isinstance(extra, Mapping):
            return None
        metadata = extra.get(_OPTIONS_METADATA)
        return metadata if isinstance(metadata, Mapping) else None

    def _resolve(self, path: str) -> tuple[Any, FieldInfo]:
        if type(path) is not str or not path:
            raise ValueError("option path must be a non-empty dotted string")
        current: OptionsConfig = self
        parts = path.split(".")
        if any(not part.isidentifier() for part in parts):
            raise ValueError(f"invalid option path {path!r}")
        for part in parts[:-1]:
            field = type(current).model_fields.get(part)
            value = getattr(current, part, None)
            if field is None or not isinstance(value, OptionsConfig):
                raise KeyError(f"unknown option path {path!r}")
            current = value
        name = parts[-1]
        field = type(current).model_fields.get(name)
        if field is None:
            raise KeyError(f"unknown option path {path!r}")
        return getattr(current, name), field

    def value(self, path: str) -> Any:
        value, _field = self._resolve(path)
        return value.value if isinstance(value, Enum) else value

    def choice(self, path: str) -> str:
        value, field = self._resolve(path)
        if self._field_metadata(field) is None:
            raise TypeError(f"option path {path!r} is not a declared choice")
        return _choice_token(value, label=f"option choice {path}")

    def option_code(self, path: str) -> int:
        value, field = self._resolve(path)
        metadata = self._field_metadata(field)
        if metadata is None or "codes" not in metadata:
            raise TypeError(f"option path {path!r} has no stable device code")
        return int(metadata["codes"][_choice_token(value, label=path)])

    def _visit(self):
        def visit(config: OptionsConfig, prefix: str):
            for name, field in type(config).model_fields.items():
                path = f"{prefix}.{name}" if prefix else name
                value = getattr(config, name)
                if isinstance(value, OptionsConfig):
                    yield from visit(value, path)
                else:
                    yield path, value, self._field_metadata(field)

        return visit(self, "")

    def specialization_key(self) -> tuple[tuple[str, str, Any], ...]:
        entries: list[tuple[str, str, Any]] = []
        for path, value, metadata in self._visit():
            role = "value" if metadata is None else str(metadata["role"])
            resolved = value.value if isinstance(value, Enum) else value
            if not isinstance(resolved, (str, int, float, bool, type(None))):
                raise TypeError(
                    f"option value {path!r} must be a JSON scalar, got "
                    f"{type(resolved).__name__}"
                )
            entries.append((path, role, resolved))
        return tuple(entries)

    def required_modules(self) -> Mapping[str, tuple[str, ...]]:
        requirements: dict[str, tuple[str, ...]] = {}
        for path, value, metadata in self._visit():
            if metadata is None or metadata.get("role") != "option":
                continue
            token = _choice_token(value, label=path)
            modules = tuple(metadata["requires_modules"][token])
            if modules:
                requirements[path] = modules
        return MappingProxyType(requirements)

    def required_forcing(self) -> Mapping[str, tuple[str, ...]]:
        requirements: dict[str, tuple[str, ...]] = {}
        for path, value, metadata in self._visit():
            if metadata is None or metadata.get("role") != "forcing":
                continue
            token = _choice_token(value, label=path)
            requirements[path] = tuple(metadata["requires"][token])
        return MappingProxyType(requirements)

    def validate_backend(self, backend: str) -> None:
        for path, value, metadata in self._visit():
            if metadata is None:
                continue
            supported = metadata.get("supported_backends", {}).get(backend)
            if supported is None:
                continue
            selected = _choice_token(value, label=path)
            if selected not in supported:
                raise ValueError(
                    f"backend {backend!r} does not support option "
                    f"{path}={selected!r}; supported values are {list(supported)}"
                )

    def validate_forcing_arguments(
        self,
        parameter_names: set[str],
        arguments: Mapping[str, Any],
    ) -> None:
        rules = self._forcing_rules
        if rules is None:
            compiled = []
            for path, value, metadata in self._visit():
                if metadata is None or metadata.get("role") != "forcing":
                    continue
                selected = _choice_token(value, label=path)
                universe = frozenset(
                    name
                    for required in metadata["requires"].values()
                    for name in required
                )
                required = frozenset(metadata["requires"][selected])
                compiled.append((path, selected, universe, required))
            rules = tuple(compiled)
            self._forcing_rules = rules
        for path, selected, universe, required in rules:
            if not universe.issubset(parameter_names):
                continue
            supplied = {name for name in universe if arguments.get(name) is not None}
            if supplied != required:
                raise ValueError(
                    f"forcing arguments do not match option {path}={selected!r}; "
                    f"required={sorted(required)}, supplied={sorted(supplied)}"
                )

    def resolved_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for name in type(self).model_fields:
            value = getattr(self, name)
            if isinstance(value, OptionsConfig):
                result[name] = value.resolved_dict()
            elif isinstance(value, Enum):
                result[name] = value.value
            else:
                result[name] = value
        return result


def build_options_group(
    class_name: str,
    source_model: type,
    fields: Mapping[str, str],
    *,
    overrides: Mapping[str, FieldInfo] | None = None,
) -> type[OptionsConfig]:
    """Build one nested options group from a declarative source schema."""

    if len(set(fields.values())) != len(fields):
        raise ValueError("option group public names must be unique")
    definitions: dict[str, tuple[Any, Any]] = {}
    overrides = overrides or {}
    for source_name, public_name in fields.items():
        source = source_model.model_fields[source_name]
        definitions[public_name] = (
            source.annotation,
            overrides[source_name] if source_name in overrides else deepcopy(source),
        )
    return create_model(
        class_name,
        __base__=OptionsConfig,
        __module__=source_model.__module__,
        **definitions,
    )


def validate_option_groups(
    source_model: type,
    groups: Mapping[str, Mapping[str, str]],
) -> None:
    """Require option groups to classify every source field exactly once."""

    assigned = Counter(name for fields in groups.values() for name in fields)
    missing = set(source_model.model_fields).difference(assigned)
    duplicates = sorted(name for name, count in assigned.items() if count > 1)
    unknown = set(assigned).difference(source_model.model_fields)
    if missing or duplicates or unknown:
        raise ValueError(
            "option groups must classify every source field exactly once; "
            f"missing={sorted(missing)}, duplicates={duplicates}, "
            f"unknown={sorted(unknown)}"
        )


__all__ = [
    "ForcingOptionField",
    "OptionField",
    "OptionsConfig",
    "build_options_group",
    "validate_option_groups",
]

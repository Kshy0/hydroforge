# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Typed immutable model-option declarations."""

from __future__ import annotations

import math
import warnings
from collections import Counter
from collections.abc import Iterable, Mapping
from contextvars import ContextVar
from copy import deepcopy
from enum import Enum
from types import MappingProxyType
from typing import Any, ClassVar, Literal, NamedTuple, Self, get_args

from pydantic import (
    ConfigDict,
    Field,
    ModelWrapValidatorHandler,
    PrivateAttr,
    ValidationInfo,
    create_model,
    field_validator,
    model_validator,
)
from pydantic.fields import FieldInfo
from pydantic_core import PydanticUndefined

from hydroforge.contracts.conditions import (
    AllOf,
    Condition,
    ConditionContext,
    OptionRef,
    as_condition,
    condition_value,
    declared_condition_errors,
    values_equal,
)
from hydroforge.core.errors import user_stacklevel
from hydroforge.core.validation import HydroForgeModel, field_default

_OPTIONS_METADATA = "hydroforge_options"
_BOOL_INPUTS: ContextVar[set[str] | None] = ContextVar(
    "options_bool_inputs", default=None
)


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


def _trait_table(
    traits: Mapping[Any, Mapping[str, Any]] | None,
    *,
    choices: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if traits is None:
        return {}
    table = _choice_mapping(traits, label="option traits", value_type=object)
    if set(table) != set(choices):
        raise ValueError(
            "option traits must cover every choice exactly; "
            f"missing={sorted(set(choices).difference(table))}, "
            f"unknown={sorted(set(table).difference(choices))}"
        )
    names: set[str] | None = None
    result: dict[str, dict[str, Any]] = {}
    for choice, values in table.items():
        if not isinstance(values, Mapping):
            raise TypeError(f"option traits[{choice!r}] must be a mapping")
        if any(type(name) is not str or not name.isidentifier() for name in values):
            raise TypeError(f"option traits[{choice!r}] names must be identifiers")
        if names is None:
            names = set(values)
        elif set(values) != names:
            raise ValueError(
                f"option traits[{choice!r}] declares {sorted(values)}; every "
                f"choice must declare the same trait names {sorted(names)}"
            )
        result[choice] = {
            name: condition_value(value) for name, value in values.items()
        }
    if not names:
        raise ValueError("option traits must declare at least one trait name")
    return result


def _relevance(value: Any) -> Condition | None:
    return None if value is None else as_condition(value)


def _parameter_effect(value: Any) -> bool | Condition:
    """Normalize ``affects_parameters``: an exact bool or a condition."""

    if type(value) is bool:
        return value
    try:
        return as_condition(value)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "affects_parameters must be an exact bool or a condition"
        ) from error


def _parameter_condition(metadata: Mapping[str, Any] | None) -> Condition | None:
    effect = None if metadata is None else metadata.get("affects_parameters")
    return effect if isinstance(effect, Condition) else None


class HydroForgeOptionWarning(UserWarning):
    """An option differs from its default while it has no effect."""


ConflictEntries = tuple[tuple[Condition, str | None], ...]


def _is_pair(value: tuple) -> bool:
    return (
        len(value) == 2
        and isinstance(value[0], (Condition, OptionRef))
        and type(value[1]) is str
    )


def _conflict_entry(value: Any) -> tuple[Condition, str | None]:
    if type(value) is tuple and _is_pair(value):
        if not value[1].strip():
            raise ValueError("conflict messages must be non-empty strings")
        return as_condition(value[0]), value[1]
    return as_condition(value), None


def _conflicts(value: Any) -> ConflictEntries:
    """Normalize a condition, a ``(condition, message)`` pair or a sequence.

    A tuple is a pair when it holds a condition and a message string, and a
    sequence of entries otherwise.
    """

    if value is None:
        return ()
    try:
        if type(value) is list or (type(value) is tuple and not _is_pair(value)):
            entries = tuple(_conflict_entry(item) for item in value)
        else:
            entries = (_conflict_entry(value),)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "conflicts must be a condition, a (condition, message) pair or a "
            f"sequence of them: {error}"
        ) from error
    if not entries:
        raise ValueError("conflicts must not be empty")
    return entries


def OptionField(
    default: Any,
    *,
    codes: Mapping[Any, int],
    requires_modules: Mapping[Any, tuple[str, ...]] | None = None,
    on_missing: Literal["error", "open"] = "error",
    traits: Mapping[Any, Mapping[str, Any]] | None = None,
    supported_backends: Mapping[str, tuple[Any, ...]] | None = None,
    relevant_when: Any = None,
    affects_parameters: bool | Any = False,
    conflicts: Any = None,
    description: str,
) -> FieldInfo:
    """Declare one user-visible choice option with a stable device code.

    ``requires_modules`` maps choices to the modules they need. With
    ``on_missing="error"`` (the default) selecting such a choice while one of
    its modules is not opened is a construction error; ``"open"`` adds the
    modules (and their required references) to ``opened_modules`` instead.
    ``traits`` maps every choice to the same named scalar traits, read by
    ``opt(path).trait(name)``. ``relevant_when`` is a condition (option paths
    relative to the declaring record) under which a non-default value has an
    effect; ``affects_parameters`` marks options that parameter or
    initial-condition files depend on: ``True``, or a condition (paths
    relative to the declaring record) under which they do, e.g.
    ``opt("/core.frozen_soil")`` or ``module("lakes")``.

    ``conflicts`` declares conditions the selected choice cannot be combined
    with: a condition, a ``(condition, message)`` pair or a sequence of them
    applies to every non-default choice; a mapping ``{choice: <same forms>}``
    gives each listed choice its own conflicts.
    """

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
    if on_missing not in {"error", "open"}:
        raise ValueError("on_missing must be 'error' or 'open'")
    complete_requirements = {
        choice: tuple(requirements.get(choice, ())) for choice in normalized_codes
    }
    if isinstance(conflicts, Mapping):
        choice_conflicts = {
            choice: _conflicts(entries)
            for choice, entries in _choice_mapping(
                conflicts, label="option conflicts", value_type=object
            ).items()
        }
        unknown = set(choice_conflicts).difference(normalized_codes)
        if unknown:
            raise ValueError(
                f"option conflicts contain unknown choices: {sorted(unknown)}"
            )
    else:
        entries = _conflicts(conflicts)
        choice_conflicts = (
            {choice: entries for choice in normalized_codes if choice != default_token}
            if entries
            else {}
        )
    return Field(
        default=default,
        description=description,
        json_schema_extra={
            _OPTIONS_METADATA: {
                "role": "option",
                "codes": normalized_codes,
                "requires_modules": complete_requirements,
                "on_missing": on_missing,
                "traits": _trait_table(traits, choices=normalized_codes),
                "supported_backends": _backend_support(
                    supported_backends,
                    choices=normalized_codes,
                ),
                "relevant_when": _relevance(relevant_when),
                "affects_parameters": _parameter_effect(affects_parameters),
                "conflicts": choice_conflicts,
            }
        },
    )


def option(
    default: Any = PydanticUndefined,
    *,
    conflicts: Any = None,
    relevant_when: Any = None,
    affects_parameters: bool | Any = False,
    **field_kwargs: Any,
) -> FieldInfo:
    """Declare a plain scalar or bool option with HydroForge metadata.

    ``field_kwargs`` are passed to :func:`pydantic.Field` (``description``,
    ``ge``, ``strict``, ``default_factory`` ...). ``relevant_when`` and
    ``affects_parameters`` have the same meaning as for :func:`OptionField`.
    ``conflicts`` (a condition, a ``(condition, message)`` pair or a sequence
    of them) applies while the option is active: an exact bool ``True``, or a
    non-bool value other than its default and ``None``.
    """

    extra = field_kwargs.pop("json_schema_extra", None)
    if extra is not None and not isinstance(extra, Mapping):
        raise TypeError("option() json_schema_extra must be a mapping")
    return Field(
        default,
        json_schema_extra={
            **(extra or {}),
            _OPTIONS_METADATA: {
                "role": "value",
                "relevant_when": _relevance(relevant_when),
                "affects_parameters": _parameter_effect(affects_parameters),
                "conflicts": _conflicts(conflicts),
            },
        },
        **field_kwargs,
    )


def ForcingOptionField(
    default: Any,
    *,
    codes: Mapping[Any, int],
    requires: Mapping[Any, tuple[str, ...]],
    supported_backends: Mapping[str, tuple[Any, ...]] | None = None,
    description: str,
) -> FieldInfo:
    """Declare one forcing-input option and its exact required arguments.

    Superseded by forcing field conditions: declare each forcing alternative
    as a forcing ``TensorField`` with ``depends_on=opt(...)`` (or ``~opt``)
    instead. Kept for existing models.
    """

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


class OptionRequirement(NamedTuple):
    """Modules required by the selected choice of one ``OptionField``."""

    path: str
    choice: str
    modules: tuple[str, ...]
    on_missing: Literal["error", "open"]


class OptionConflict(NamedTuple):
    """One declared conflict of an active option, with an absolute condition."""

    path: str
    value: Any
    condition: Condition
    message: str | None


class OptionRelevance(NamedTuple):
    """One stored option with its default and its absolute relevance condition."""

    path: str
    value: Any
    default: Any
    condition: Condition | None


def reject_rules_declaration(cls: type) -> None:
    """Fail loudly on the removed ``__rules__`` declaration."""

    if "__rules__" in cls.__dict__:
        raise TypeError(
            f"{cls.__name__}.__rules__ is no longer supported; declare "
            "option(..., conflicts=...) / OptionField(..., conflicts=...) or "
            "check combinations in an option or module validator"
        )


def _record_relevance(cls: type) -> Condition | None:
    value = getattr(cls, "__relevant_when__", None)
    return None if value is None else as_condition(value)


class OptionsConfig(HydroForgeModel):
    """Immutable nested model options, separate from tensor input.

    ``__relevant_when__`` is a condition, with paths relative to this record,
    applied to every option stored in it; it is evaluated once at model
    construction.
    """

    model_config = ConfigDict(strict=False)
    __relevant_when__: ClassVar[Any] = None
    _coerced_bool_fields: frozenset[str] = PrivateAttr(default=frozenset())

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        super().__pydantic_init_subclass__(**kwargs)
        reject_rules_declaration(cls)
        if cls.__pydantic_complete__:
            errors = cls.declaration_errors()
            if errors:
                raise ValueError(
                    f"{cls.__name__} declares invalid option contracts:\n  - "
                    + "\n  - ".join(errors)
                )

    @classmethod
    def referenced_modules(cls) -> frozenset[str]:
        """Return module names named by this record type and nested records.

        Covers ``requires_modules`` of every choice and every declared
        condition, so a model can reject unknown names when its class is
        defined.
        """

        names: set[str] = set()
        seen: set[type] = set()
        pending: list[type] = [cls]
        while pending:
            record = pending.pop()
            if record in seen:
                continue
            seen.add(record)
            for condition in record._declared_conditions():
                names.update(condition.modules())
            for field in record.model_fields.values():
                metadata = cls._field_metadata(field)
                if metadata is not None:
                    for modules in metadata.get("requires_modules", {}).values():
                        names.update(modules)
                pending.extend(
                    item
                    for item in (get_args(field.annotation) or (field.annotation,))
                    if isinstance(item, type) and issubclass(item, OptionsConfig)
                )
        return frozenset(names)

    @classmethod
    def _declared_conditions(cls) -> list[Condition]:
        """Return this record's declared conditions (record-relative paths)."""

        relevance = _record_relevance(cls)
        conditions = [] if relevance is None else [relevance]
        for field in cls.model_fields.values():
            metadata = cls._field_metadata(field)
            if metadata is None:
                continue
            for condition in (
                metadata.get("relevant_when"),
                _parameter_condition(metadata),
            ):
                if condition is not None:
                    conditions.append(condition)
            entries = metadata.get("conflicts", ())
            if isinstance(entries, Mapping):
                entries = tuple(entry for group in entries.values() for entry in group)
            conditions.extend(condition for condition, _message in entries)
        return conditions

    @classmethod
    def declaration_errors(cls) -> list[str]:
        """Check this record's declared conditions against its fields.

        Root-relative ``"/..."`` paths are checked by
        :meth:`nested_declaration_errors` of the root type instead.
        """

        try:
            conditions = cls._declared_conditions()
        except (TypeError, ValueError) as error:
            return [str(error)]
        errors: list[str] = []
        for condition in conditions:
            errors.extend(declared_condition_errors(condition, cls, root=False))
        return list(dict.fromkeys(errors))

    @classmethod
    def nested_declaration_errors(cls) -> list[str]:
        """Check every nested record's declarations against this root type.

        Called when a model class is defined with this as its ``options``
        type, the first point at which root-relative paths can be checked.
        """

        errors: list[str] = []

        def visit(record: type[OptionsConfig], prefix: str, stack: tuple) -> None:
            try:
                conditions = record._declared_conditions()
            except (TypeError, ValueError):
                return  # reported when the record class was defined
            for condition in conditions:
                errors.extend(
                    declared_condition_errors(condition.prefixed(prefix), cls)
                )
            for name, field in record.model_fields.items():
                for item in get_args(field.annotation) or (field.annotation,):
                    if (
                        isinstance(item, type)
                        and issubclass(item, OptionsConfig)
                        and item not in stack
                    ):
                        path = f"{prefix}.{name}" if prefix else name
                        visit(item, path, (*stack, item))

        visit(cls, "", (cls,))
        return list(dict.fromkeys(errors))

    @model_validator(mode="wrap")
    @classmethod
    def _remember_bool_inputs(
        cls, value: Any, handler: ModelWrapValidatorHandler
    ) -> Self:
        # Keep legacy option coercion. Only options later used for field
        # activation reject coerced Boolean inputs; nested records have their
        # own construction-local collector (also safe for concurrent models).
        if isinstance(value, cls):
            return handler(value)
        coerced: set[str] = set()
        token = _BOOL_INPUTS.set(coerced)
        try:
            result = handler(value)
        finally:
            _BOOL_INPUTS.reset(token)
        result._coerced_bool_fields = frozenset(coerced)
        return result

    @field_validator("*", mode="before")
    @classmethod
    def _record_bool_input(cls, value: Any, info: ValidationInfo) -> Any:
        collector = _BOOL_INPUTS.get()
        if (
            collector is not None
            and cls.model_fields[info.field_name].annotation is bool
            and type(value) is not bool
        ):
            collector.add(info.field_name)
        return value

    _forcing_rules: (
        tuple[tuple[str, str, frozenset[str], frozenset[str]], ...] | None
    ) = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _validate_declared_choices(self) -> Self:
        # Pydantic reports only ValueError/AssertionError as ValidationError.
        try:
            for name, field in type(self).model_fields.items():
                metadata = self._field_metadata(field)
                if metadata is None or "codes" not in metadata:
                    continue
                token = _choice_token(getattr(self, name), label=f"option field {name}")
                choices = metadata["codes"]
                if token not in choices:
                    raise ValueError(
                        f"option field {name!r} has unsupported choice {token!r}; "
                        f"expected one of {sorted(choices)}"
                    )
            self.specialization_key()
        except TypeError as error:
            raise ValueError(str(error)) from error
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
        metadata = self._field_metadata(field)
        if metadata is None or "codes" not in metadata:
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
            if type(resolved) not in {str, int, float, bool, type(None)}:
                raise TypeError(
                    f"option value {path!r} must be a JSON scalar, got "
                    f"{type(resolved).__name__}"
                )
            if type(resolved) is float and not math.isfinite(resolved):
                raise ValueError(
                    f"option value {path!r} must be finite, got {resolved!r}"
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

    def option_requirements(self) -> tuple[OptionRequirement, ...]:
        """Return the module requirements of every selected choice."""

        requirements = []
        for path, value, metadata in self._visit():
            if metadata is None or metadata.get("role") != "option":
                continue
            token = _choice_token(value, label=path)
            modules = tuple(metadata["requires_modules"][token])
            if modules:
                requirements.append(
                    OptionRequirement(
                        path, token, modules, metadata.get("on_missing", "error")
                    )
                )
        return tuple(requirements)

    def conflicts(self) -> tuple[OptionConflict, ...]:
        """Return the declared conflicts of every active option.

        A plain option is active when it holds an exact bool ``True`` or a
        non-bool value other than its default and ``None``; an
        ``OptionField`` when its selected choice declares conflicts.
        """

        entries: list[OptionConflict] = []

        def visit(record: OptionsConfig, prefix: str) -> None:
            for name, field in type(record).model_fields.items():
                path = f"{prefix}.{name}" if prefix else name
                value = getattr(record, name)
                if isinstance(value, OptionsConfig):
                    visit(value, path)
                    continue
                metadata = self._field_metadata(field)
                declared = None if metadata is None else metadata.get("conflicts")
                if not declared:
                    continue
                if isinstance(declared, Mapping):
                    selected = declared.get(_choice_token(value, label=path), ())
                elif type(value) is bool:
                    selected = declared if value else ()
                elif value is None or _same_value(value, field_default(field)):
                    selected = ()
                else:
                    selected = declared
                entries.extend(
                    OptionConflict(path, value, condition.prefixed(prefix), message)
                    for condition, message in selected
                )

        visit(self, "")
        return tuple(entries)

    def relevance(self) -> tuple[OptionRelevance, ...]:
        """Return every stored option with its absolute relevance condition.

        The condition is the AND of the option's ``relevant_when`` and the
        ``__relevant_when__`` of every enclosing record, or ``None``.
        """

        entries: list[OptionRelevance] = []

        def visit(
            record: OptionsConfig, prefix: str, inherited: tuple[Condition, ...]
        ) -> None:
            own = _record_relevance(type(record))
            if own is not None:
                inherited = (*inherited, own.prefixed(prefix))
            for name, field in type(record).model_fields.items():
                path = f"{prefix}.{name}" if prefix else name
                value = getattr(record, name)
                if isinstance(value, OptionsConfig):
                    visit(value, path, inherited)
                    continue
                metadata = self._field_metadata(field)
                local = None if metadata is None else metadata.get("relevant_when")
                conditions = inherited
                if local is not None:
                    conditions = (*conditions, local.prefixed(prefix))
                entries.append(
                    OptionRelevance(
                        path,
                        value,
                        field_default(field),
                        AllOf.of(*conditions) if conditions else None,
                    )
                )

        visit(self, "", ())
        return tuple(entries)

    def parameter_option_paths(self) -> frozenset[str]:
        """Return every stored option declaring ``affects_parameters``.

        Includes options whose ``affects_parameters`` condition is false.
        """

        return frozenset(
            path
            for path, _value, metadata in self._visit()
            if metadata is not None
            and (
                metadata.get("affects_parameters") is True
                or _parameter_condition(metadata) is not None
            )
        )

    def parameter_options(
        self, *, opened_modules: Iterable[str] | None = None
    ) -> Mapping[str, Any]:
        """Return ``{path: value}`` of options that parameter files depend on.

        An option declared ``affects_parameters=True`` is always included; one
        declared with a condition only while it holds for these options and
        ``opened_modules``. Without ``opened_modules``, a condition that names
        modules cannot be decided and its option is kept. With
        ``opened_modules``, options whose ``relevant_when`` is false are also
        left out. An option whose condition cannot be evaluated is kept.
        """

        context = ConditionContext(frozenset(opened_modules or ()), options=self)

        def holds(condition: Condition | None) -> bool:
            if condition is None:
                return True
            try:
                return condition.evaluate(context)
            except (TypeError, ValueError):
                return True

        declared = self.parameter_option_paths()
        values: dict[str, Any] = {}
        for path, value, metadata in self._visit():
            if path not in declared:
                continue
            condition = _parameter_condition(metadata)
            if condition is not None:
                condition = condition.prefixed(path.rpartition(".")[0])
                if (opened_modules is not None or not condition.modules()) and not (
                    holds(condition)
                ):
                    continue
            values[path] = value.value if isinstance(value, Enum) else value
        if opened_modules is not None:
            for entry in self.relevance():
                if entry.path in values and not holds(entry.condition):
                    del values[entry.path]
        return MappingProxyType(values)

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
        # Frozen HydroForge defaults are immutable and refuse deep copies.
        memo = (
            {id(source.default): source.default}
            if isinstance(source.default, HydroForgeModel)
            else {}
        )
        definitions[public_name] = (
            source.annotation,
            overrides[source_name]
            if source_name in overrides
            else deepcopy(source, memo),
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


def _same_value(value: Any, default: Any) -> bool:
    try:
        return values_equal(value, default)
    except (TypeError, ValueError):
        return value == default


def _display(value: Any) -> str:
    try:
        return repr(condition_value(value))
    except (TypeError, ValueError):
        return repr(value)


def check_option_contracts(
    options: OptionsConfig,
    opened_modules: Iterable[str],
    *,
    strict_options: bool = False,
) -> None:
    """Validate one model's options against its module selection.

    Checks modules required by selected choices (``on_missing="error"``) and
    the ``conflicts`` of active options, and collects options whose
    ``relevant_when`` is false. Every violation raises together as one
    ``ValueError``; irrelevant options warn with
    :class:`HydroForgeOptionWarning`, or are violations with
    ``strict_options``.
    """

    opened = tuple(opened_modules)
    context = ConditionContext(frozenset(opened), options=options)
    problems: list[str] = []
    for requirement in options.option_requirements():
        missing = [name for name in requirement.modules if name not in opened]
        if missing and requirement.on_missing == "error":
            problems.append(
                f"options.{requirement.path}={requirement.choice!r} requires "
                f"modules {missing}, which are not in opened_modules; open them "
                "or declare the option with on_missing='open'"
            )
    for entry in options.conflicts():
        try:
            conflicting = entry.condition.evaluate(context)
        except (TypeError, ValueError) as error:
            problems.append(
                f"conflicts of options.{entry.path} cannot be evaluated: {error}"
            )
            continue
        if conflicting:
            text = (
                f"options.{entry.path}={_display(entry.value)} conflicts with "
                f"{entry.condition}"
            )
            problems.append(f"{text}: {entry.message}" if entry.message else text)
    irrelevant = []
    for entry in options.relevance():
        if entry.condition is None or _same_value(entry.value, entry.default):
            continue
        try:
            relevant = entry.condition.evaluate(context)
        except (TypeError, ValueError) as error:
            problems.append(
                f"relevance of options.{entry.path} cannot be evaluated: {error}"
            )
            continue
        if not relevant:
            irrelevant.append(
                f"options.{entry.path}={_display(entry.value)} "
                f"has no effect unless {entry.condition}"
            )
    if irrelevant and strict_options:
        problems.extend(irrelevant)
    elif irrelevant:
        warnings.warn(
            "options differ from their defaults but have no effect: "
            + "; ".join(irrelevant),
            HydroForgeOptionWarning,
            stacklevel=user_stacklevel(),
        )
    if problems:
        raise ValueError(
            "model options violate declared contracts:\n  - " + "\n  - ".join(problems)
        )


__all__ = [
    "ForcingOptionField",
    "HydroForgeOptionWarning",
    "OptionConflict",
    "OptionField",
    "OptionRelevance",
    "OptionRequirement",
    "OptionsConfig",
    "build_options_group",
    "check_option_contracts",
    "option",
    "validate_option_groups",
]

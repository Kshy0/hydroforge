# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Declarative conditions over opened modules and declared model options.

A :class:`Condition` is a frozen, hashable and picklable expression tree built
from :func:`module`, :func:`opt`, :data:`always` and :data:`never` with the
``&``, ``|`` and ``~`` operators. Option paths are relative to the model's
root ``options`` (or, inside an ``OptionsConfig`` record's declarations, to
that record) and are read through declared, frozen ``OptionsConfig`` records
only: properties, runtime state, expressions and tensors are never evaluated.

Legacy spellings coerce through :func:`as_condition`: ``"name"`` is
``module("name")``, ``"options.a.b"`` is ``opt("a.b")`` (an exact-bool flag)
and a tuple is the AND of its items.
"""

from __future__ import annotations

import inspect
import math
import operator
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, get_args

from pydantic_core import core_schema

if TYPE_CHECKING:
    from hydroforge.contracts.options import OptionsConfig

_SCALAR_TYPES = (str, int, float, bool, type(None))
_ORDERINGS: Mapping[str, Callable[[Any, Any], bool]] = MappingProxyType(
    {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
)
_COERCION_ERROR = (
    "conditions must be a module identifier, an options.<public path> string, "
    "a tuple of them (AND), or a Condition"
)


ROOT = "/"
"""Leading marker of a root-relative option path inside an options record."""


def _option_path(value: Any) -> str:
    if type(value) is not str or not value:
        raise TypeError("option paths must be non-empty dotted strings")
    if any(
        not part.isidentifier() or part.startswith("_")
        for part in value.removeprefix(ROOT).split(".")
    ):
        raise ValueError(
            f"option path {value!r} must be a dotted path of public identifiers, "
            f"optionally starting with {ROOT!r} (root-relative)"
        )
    return value


def is_root_path(path: str) -> bool:
    """Whether ``path`` is root-relative (``"/a.b"``) inside a record."""

    return path.startswith(ROOT)


def prefix_path(path: str, prefix: str) -> str:
    """Rebase a record-relative path beneath ``prefix``; root paths drop ``/``."""

    if is_root_path(path):
        return path.removeprefix(ROOT)
    return f"{prefix}.{path}" if prefix else path


def _shown(path: str) -> str:
    return f"options.{path.removeprefix(ROOT)}"


def condition_value(value: Any) -> Any:
    """Return the comparable scalar of an option value (Enum members by value)."""

    raw = value.value if isinstance(value, Enum) else value
    if type(raw) not in _SCALAR_TYPES:
        raise TypeError(
            "condition values must be str, int, float, bool or None (or Enum "
            f"members of them), got {type(raw).__name__}"
        )
    if type(raw) is float and not math.isfinite(raw):
        raise ValueError(f"condition values must be finite, got {raw!r}")
    return raw


def values_equal(left: Any, right: Any) -> bool:
    """Equality without bool/number cross-matching (``True != 1`` here)."""

    left, right = condition_value(left), condition_value(right)
    if (type(left) is bool) != (type(right) is bool):
        return False
    return bool(left == right)


def _typed(value: Any) -> tuple[bool, Any]:
    """Equality key of a condition value: ``True`` never equals ``1``."""

    return type(value) is bool, value


class Condition:
    """Abstract boolean condition over opened modules and model options."""

    __slots__ = ()

    def evaluate(self, context: ConditionContext) -> bool:
        """Return whether the condition holds in ``context``."""

        raise NotImplementedError

    def modules(self) -> frozenset[str]:
        """Return every module name this condition reads."""

        return frozenset()

    def option_atoms(self) -> tuple[OptionAtom, ...]:
        """Return every option-reading leaf, in declaration order."""

        return ()

    def option_paths(self) -> frozenset[str]:
        """Return every option path (relative, dotted) this condition reads."""

        return frozenset(path for atom in self.option_atoms() for path in atom.paths())

    def prefixed(self, prefix: str) -> Condition:
        """Return this condition with option paths rebased beneath ``prefix``."""

        del prefix
        return self

    def __and__(self, other: Any) -> Condition:
        return AllOf.of(self, as_condition(other))

    def __rand__(self, other: Any) -> Condition:
        return AllOf.of(as_condition(other), self)

    def __or__(self, other: Any) -> Condition:
        return AnyOf.of(self, as_condition(other))

    def __ror__(self, other: Any) -> Condition:
        return AnyOf.of(as_condition(other), self)

    def __invert__(self) -> Condition:
        return Not(self)

    def __bool__(self) -> bool:
        raise TypeError(
            "conditions have no truth value; combine them with &, | and ~ "
            "and evaluate them with .evaluate(context)"
        )

    def __str__(self) -> str:
        raise NotImplementedError

    @classmethod
    def __get_pydantic_core_schema__(
        cls, source: Any, handler: Any
    ) -> core_schema.CoreSchema:
        del source, handler
        return core_schema.no_info_plain_validator_function(
            as_condition,
            serialization=core_schema.plain_serializer_function_ser_schema(str),
        )


@dataclass(frozen=True, slots=True)
class ConditionContext:
    """What a condition may read: opened modules and the model options.

    ``resolved`` optionally maps ``str(atom)`` to the exact bool resolved for
    an option atom earlier (a model plan's ``conditions``); atoms absent from
    it are read from ``options``.
    """

    modules: frozenset[str]
    options: OptionsConfig | None = None
    resolved: Mapping[str, bool] | None = None

    def __post_init__(self) -> None:
        if isinstance(self.modules, str):
            raise TypeError("condition context modules must be an iterable of names")
        object.__setattr__(self, "modules", frozenset(self.modules))

    def option(self, atom: OptionAtom) -> bool:
        """Return the value of one option atom."""

        if self.resolved is not None:
            key = str(atom)
            if key in self.resolved:
                value = self.resolved[key]
                if type(value) is not bool:
                    raise TypeError(f"resolved condition {key!r} must be an exact bool")
                return value
        if self.options is None:
            raise ValueError(
                f"unresolved options condition {str(atom)!r}; provide an options context"
            )
        return atom.resolve(self.options)


@dataclass(frozen=True, slots=True)
class Constant(Condition):
    """``always`` or ``never``."""

    value: bool

    def evaluate(self, context: ConditionContext) -> bool:
        del context
        return self.value

    def __str__(self) -> str:
        return "always" if self.value else "never"


always = Constant(True)
never = Constant(False)


@dataclass(frozen=True, slots=True)
class ModuleOpen(Condition):
    """The named module is opened."""

    name: str

    def __post_init__(self) -> None:
        if type(self.name) is not str or not self.name.isidentifier():
            raise ValueError(
                f"module conditions require an identifier, got {self.name!r}"
            )

    def evaluate(self, context: ConditionContext) -> bool:
        return self.name in context.modules

    def modules(self) -> frozenset[str]:
        return frozenset((self.name,))

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True, slots=True)
class OptionAtom(Condition):
    """A leaf that reads one declared option."""

    path: str

    def __post_init__(self) -> None:
        _option_path(self.path)

    def evaluate(self, context: ConditionContext) -> bool:
        return context.option(self)

    def option_atoms(self) -> tuple[OptionAtom, ...]:
        return (self,)

    def paths(self) -> tuple[str, ...]:
        """Return the option paths this atom reads."""

        return (self.path,)

    def prefixed(self, prefix: str) -> Condition:
        path = prefix_path(self.path, prefix)
        return self if path == self.path else replace(self, path=path)

    def resolve(self, options: OptionsConfig) -> bool:
        """Read this atom strictly from declared options."""

        raise NotImplementedError

    def check_declared(self, options_type: type[OptionsConfig]) -> None:
        """Reject a path that the declared options type cannot hold."""

        declared_option(options_type, self.path, condition=str(self))

    def _read(self, options: OptionsConfig) -> tuple[Any, Any]:
        return read_option(options, self.path, condition=str(self))


@dataclass(frozen=True, slots=True)
class OptionFlag(OptionAtom):
    """A declared exact-bool option is ``True`` (coerced input is rejected)."""

    def resolve(self, options: OptionsConfig) -> bool:
        value, field = self._read(options)
        if field.annotation is not bool or type(value) is not bool:
            raise ValueError(f"condition {str(self)!r} requires a declared exact bool")
        record, name = _owner(options, self.path)
        if name in record._coerced_bool_fields:
            raise ValueError(
                f"condition {str(self)!r} requires exact bool input, not a coerced value"
            )
        return value

    def check_declared(self, options_type: type[OptionsConfig]) -> None:
        field = declared_option(options_type, self.path, condition=str(self))
        if field.annotation is not bool:
            raise ValueError(f"condition {str(self)!r} requires a declared exact bool")

    def __str__(self) -> str:
        return _shown(self.path)


def _choice_errors(field: Any, values: Iterable[Any], condition: str) -> None:
    from hydroforge.contracts.options import OptionsConfig

    metadata = OptionsConfig._field_metadata(field)
    if metadata is None or "codes" not in metadata:
        return
    unknown = [
        value
        for value in values
        if type(value) is not str or value not in metadata["codes"]
    ]
    if unknown:
        raise ValueError(
            f"condition {condition!r} compares with unknown choices {unknown}; "
            f"expected one of {sorted(metadata['codes'])}"
        )


@dataclass(frozen=True, slots=True)
class OptionCompare(OptionAtom):
    """``opt(path) <op> value``; Enum members compare by ``.value``."""

    op: str
    value: Any

    def __post_init__(self) -> None:
        _option_path(self.path)
        if self.op not in {"==", "!=", *_ORDERINGS}:
            raise ValueError(f"unsupported option comparison {self.op!r}")
        object.__setattr__(self, "value", condition_value(self.value))
        if self.op in _ORDERINGS and type(self.value) not in {int, float}:
            raise TypeError(
                f"option ordering {self.op!r} requires an int or float, "
                f"got {self.value!r}"
            )

    def _key(self) -> tuple[Any, ...]:
        return self.path, self.op, _typed(self.value)

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def resolve(self, options: OptionsConfig) -> bool:
        actual = condition_value(self._read(options)[0])
        if self.op == "==":
            return values_equal(actual, self.value)
        if self.op == "!=":
            return not values_equal(actual, self.value)
        if type(actual) not in {int, float}:
            raise ValueError(
                f"condition {str(self)!r} requires a numeric option, got {actual!r}"
            )
        return _ORDERINGS[self.op](actual, self.value)

    def check_declared(self, options_type: type[OptionsConfig]) -> None:
        field = declared_option(options_type, self.path, condition=str(self))
        if self.op in {"==", "!="}:
            _choice_errors(field, (self.value,), str(self))

    def __str__(self) -> str:
        return f"{_shown(self.path)} {self.op} {self.value!r}"


@dataclass(frozen=True, slots=True)
class OptionPairCompare(OptionAtom):
    """``opt(path) <op> opt(other)``: two options compared with each other."""

    op: str
    other: str

    def __post_init__(self) -> None:
        _option_path(self.path)
        _option_path(self.other)
        if self.op not in {"==", "!=", *_ORDERINGS}:
            raise ValueError(f"unsupported option comparison {self.op!r}")

    def paths(self) -> tuple[str, ...]:
        return (self.path, self.other)

    def prefixed(self, prefix: str) -> Condition:
        return replace(
            self,
            path=prefix_path(self.path, prefix),
            other=prefix_path(self.other, prefix),
        )

    def resolve(self, options: OptionsConfig) -> bool:
        left = condition_value(self._read(options)[0])
        right = condition_value(
            read_option(options, self.other, condition=str(self))[0]
        )
        if self.op == "==":
            return values_equal(left, right)
        if self.op == "!=":
            return not values_equal(left, right)
        if type(left) not in {int, float} or type(right) not in {int, float}:
            raise ValueError(
                f"condition {str(self)!r} requires numeric options, got "
                f"{left!r} and {right!r}"
            )
        return _ORDERINGS[self.op](left, right)

    def check_declared(self, options_type: type[OptionsConfig]) -> None:
        for path in self.paths():
            declared_option(options_type, path, condition=str(self))

    def __str__(self) -> str:
        return f"{_shown(self.path)} {self.op} {_shown(self.other)}"


@dataclass(frozen=True, slots=True)
class OptionIn(OptionAtom):
    """``opt(path).in_(*values)``."""

    values: tuple[Any, ...]

    def __post_init__(self) -> None:
        _option_path(self.path)
        if type(self.values) is not tuple or not self.values:
            raise ValueError("in_() requires at least one value")
        object.__setattr__(
            self, "values", tuple(condition_value(item) for item in self.values)
        )

    def _key(self) -> tuple[Any, ...]:
        return self.path, tuple(map(_typed, self.values))

    def __eq__(self, other: object) -> bool:
        if type(other) is not type(self):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def resolve(self, options: OptionsConfig) -> bool:
        actual = self._read(options)[0]
        return any(values_equal(actual, item) for item in self.values)

    def check_declared(self, options_type: type[OptionsConfig]) -> None:
        field = declared_option(options_type, self.path, condition=str(self))
        _choice_errors(field, self.values, str(self))

    def __str__(self) -> str:
        return f"{_shown(self.path)} in {self.values!r}"


@dataclass(frozen=True, slots=True)
class OptionTrait(OptionAtom):
    """The selected choice of an ``OptionField`` has a truthy trait."""

    trait: str

    def __post_init__(self) -> None:
        _option_path(self.path)
        if type(self.trait) is not str or not self.trait.isidentifier():
            raise ValueError(f"trait names must be identifiers, got {self.trait!r}")

    def _traits(self, field: Any) -> Mapping[str, Mapping[str, Any]]:
        from hydroforge.contracts.options import OptionsConfig

        metadata = OptionsConfig._field_metadata(field)
        traits = None if metadata is None else metadata.get("traits")
        if not traits or self.trait not in next(iter(traits.values())):
            raise ValueError(
                f"condition {str(self)!r} names a trait the option does not declare"
            )
        return traits

    def resolve(self, options: OptionsConfig) -> bool:
        value, field = self._read(options)
        traits = self._traits(field)
        token = condition_value(value)
        return bool(traits[token][self.trait])

    def check_declared(self, options_type: type[OptionsConfig]) -> None:
        self._traits(declared_option(options_type, self.path, condition=str(self)))

    def __str__(self) -> str:
        return f"{_shown(self.path)}.trait({self.trait!r})"


@dataclass(frozen=True, slots=True)
class OptionIsSet(OptionAtom):
    """The option (and every record on its path) is not ``None``."""

    def resolve(self, options: OptionsConfig) -> bool:
        value, _field = read_option(
            options, self.path, condition=str(self), none_records=True
        )
        return value is not None

    def __str__(self) -> str:
        return f"{_shown(self.path)} is set"


def _flatten(kind: type, items: Iterable[Condition]) -> tuple[Condition, ...]:
    flat: list[Condition] = []
    for item in items:
        for part in item.items if type(item) is kind else (item,):
            if part not in flat:
                flat.append(part)
    return tuple(flat)


@dataclass(frozen=True, slots=True)
class _Compound(Condition):
    items: tuple[Condition, ...]
    symbol: ClassVar[str] = ""

    @classmethod
    def of(cls, *items: Condition) -> Condition:
        flat = _flatten(cls, items)
        return flat[0] if len(flat) == 1 else cls(flat)

    def modules(self) -> frozenset[str]:
        return frozenset().union(*(item.modules() for item in self.items))

    def option_atoms(self) -> tuple[OptionAtom, ...]:
        return tuple(atom for item in self.items for atom in item.option_atoms())

    def prefixed(self, prefix: str) -> Condition:
        return type(self)(tuple(item.prefixed(prefix) for item in self.items))

    def __str__(self) -> str:
        return "(" + f" {self.symbol} ".join(map(str, self.items)) + ")"


@dataclass(frozen=True, slots=True)
class AllOf(_Compound):
    """Every item holds."""

    symbol: ClassVar[str] = "&"

    def evaluate(self, context: ConditionContext) -> bool:
        return all(item.evaluate(context) for item in self.items)


@dataclass(frozen=True, slots=True)
class AnyOf(_Compound):
    """At least one item holds."""

    symbol: ClassVar[str] = "|"

    def evaluate(self, context: ConditionContext) -> bool:
        return any(item.evaluate(context) for item in self.items)


@dataclass(frozen=True, slots=True)
class Not(Condition):
    """The item does not hold."""

    item: Condition

    def evaluate(self, context: ConditionContext) -> bool:
        return not self.item.evaluate(context)

    def modules(self) -> frozenset[str]:
        return self.item.modules()

    def option_atoms(self) -> tuple[OptionAtom, ...]:
        return self.item.option_atoms()

    def prefixed(self, prefix: str) -> Condition:
        return Not(self.item.prefixed(prefix))

    def __invert__(self) -> Condition:
        return self.item

    def __str__(self) -> str:
        inner = str(self.item)
        return (
            f"~{inner}"
            if isinstance(self.item, (ModuleOpen, OptionFlag, _Compound))
            else f"~({inner})"
        )


class OptionRef:
    """A reference to one option path; comparisons build conditions.

    Used directly as a condition, it means "this declared exact bool option
    is ``True``". It is deliberately unhashable because ``==`` builds a
    condition instead of comparing references.
    """

    __slots__ = ("path",)
    __hash__ = None  # type: ignore[assignment]

    def __init__(self, path: str) -> None:
        object.__setattr__(self, "path", _option_path(path))

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("OptionRef is immutable")

    def __reduce__(self) -> tuple[Any, ...]:
        return (OptionRef, (self.path,))

    def __eq__(self, value: Any) -> Condition:  # type: ignore[override]
        return self._compare("==", value)

    def __ne__(self, value: Any) -> Condition:  # type: ignore[override]
        return self._compare("!=", value)

    def __lt__(self, value: Any) -> Condition:
        return self._compare("<", value)

    def __le__(self, value: Any) -> Condition:
        return self._compare("<=", value)

    def __gt__(self, value: Any) -> Condition:
        return self._compare(">", value)

    def __ge__(self, value: Any) -> Condition:
        return self._compare(">=", value)

    def _compare(self, op: str, value: Any) -> Condition:
        if isinstance(value, OptionRef):
            return OptionPairCompare(self.path, op, value.path)
        return OptionCompare(self.path, op, value)

    def in_(self, *values: Any) -> Condition:
        """The option equals one of ``values``."""

        return OptionIn(self.path, values)

    def trait(self, name: str) -> Condition:
        """The selected choice's declared trait ``name`` is truthy."""

        return OptionTrait(self.path, name)

    def is_set(self) -> Condition:
        """The option is not ``None`` (and no record on its path is ``None``)."""

        return OptionIsSet(self.path)

    def as_condition(self) -> Condition:
        return OptionFlag(self.path)

    def __and__(self, other: Any) -> Condition:
        return self.as_condition() & other

    def __rand__(self, other: Any) -> Condition:
        return as_condition(other) & self.as_condition()

    def __or__(self, other: Any) -> Condition:
        return self.as_condition() | other

    def __ror__(self, other: Any) -> Condition:
        return as_condition(other) | self.as_condition()

    def __invert__(self) -> Condition:
        return Not(self.as_condition())

    def __bool__(self) -> bool:
        raise TypeError(
            "option references have no truth value; use them as conditions or "
            "compare them with ==, !=, <, <=, >, >=, .in_(), .trait() or .is_set()"
        )

    def __str__(self) -> str:
        return _shown(self.path)

    def __repr__(self) -> str:
        return f"opt({self.path!r})"


def module(name: str) -> Condition:
    """The named module is opened."""

    return ModuleOpen(name)


def opt(path: str) -> OptionRef:
    """Reference an option by its dotted path.

    Paths are relative to the root ``options`` in module fields, and relative
    to the declaring record in ``OptionsConfig`` declarations
    (``conflicts``, ``relevant_when``, ``affects_parameters``,
    ``__relevant_when__``).
    A leading ``"/"`` (``opt("/core.full_energy")``) makes a path
    root-relative everywhere.
    """

    return OptionRef(path)


def as_condition(value: Any) -> Condition:
    """Coerce a condition or one of its legacy spellings."""

    if isinstance(value, Condition):
        return value
    if isinstance(value, OptionRef):
        return value.as_condition()
    if type(value) is str:
        if value.isidentifier():
            return ModuleOpen(value)
        if value.startswith("options."):
            try:
                return OptionFlag(_option_path(value.removeprefix("options.")))
            except (TypeError, ValueError):
                pass
        raise ValueError(f"{_COERCION_ERROR}; got {value!r}")
    if type(value) is tuple:
        if not value:
            return always
        return AllOf.of(*(as_condition(item) for item in value))
    raise TypeError(f"{_COERCION_ERROR}; got {type(value).__name__}")


def _record_types(annotation: Any) -> tuple[Any, ...]:
    args = get_args(annotation) or (annotation,)
    return tuple(item for item in args if item is not type(None))


def _is_record_type(item: Any) -> bool:
    from hydroforge.contracts.options import OptionsConfig

    return isinstance(item, type) and issubclass(item, OptionsConfig)


def read_option(
    options: OptionsConfig,
    path: str,
    *,
    condition: str,
    none_records: bool = False,
) -> tuple[Any, Any]:
    """Read one stored option and its ``FieldInfo`` through declared records.

    With ``none_records=True`` a ``None`` record on the path reads as
    ``(None, None)`` instead of failing.
    """

    from hydroforge.contracts.options import OptionsConfig

    current: Any = options
    parts = path.removeprefix(ROOT).split(".")
    for index, part in enumerate(parts):
        if current is None:
            if none_records:
                return None, None
            record = ".".join(("options", *parts[:index]))
            raise ValueError(
                f"condition {condition!r} traverses {record!r}, which is None; "
                "set that options record or remove the condition"
            )
        if not isinstance(current, OptionsConfig) or not type(current).model_config.get(
            "frozen"
        ):
            raise ValueError(
                f"condition {condition!r} requires frozen OptionsConfig records"
            )
        field = type(current).model_fields.get(part)
        if field is None:
            raise ValueError(
                f"condition {condition!r} names an unknown declared option {part!r}"
            )
        descriptor = inspect.getattr_static(type(current), part, None)
        if descriptor is not None and hasattr(descriptor, "__get__"):
            raise ValueError(f"condition {condition!r} cannot traverse descriptors")
        # Read stored declared fields directly, never __getattr__/properties.
        values = object.__getattribute__(current, "__dict__")
        if part not in values:
            raise ValueError(
                f"condition {condition!r} has no stored value for {part!r}"
            )
        value = values[part]
        if index == len(parts) - 1:
            return value, field
        records = _record_types(field.annotation)
        if not records or not all(_is_record_type(item) for item in records):
            raise ValueError(
                f"condition {condition!r} requires a declared OptionsConfig path"
            )
        current = value
    raise ValueError(f"empty condition {condition!r}")


def _owner(options: OptionsConfig, path: str) -> tuple[OptionsConfig, str]:
    *records, name = path.removeprefix(ROOT).split(".")
    current = options
    for part in records:
        current = object.__getattribute__(current, "__dict__")[part]
    return current, name


def declared_option(
    options_type: type[OptionsConfig], path: str, *, condition: str
) -> Any:
    """Return the declared ``FieldInfo`` of ``path`` in an options class."""

    current = options_type
    parts = path.removeprefix(ROOT).split(".")
    for index, part in enumerate(parts):
        field = current.model_fields.get(part)
        if field is None:
            raise ValueError(
                f"condition {condition!r} names an unknown declared option "
                f"{'.'.join(parts[: index + 1])!r} of {options_type.__name__}"
            )
        if index == len(parts) - 1:
            return field
        records = _record_types(field.annotation)
        if len(records) != 1 or not _is_record_type(records[0]):
            raise ValueError(
                f"condition {condition!r} requires a declared OptionsConfig "
                f"record at {'.'.join(parts[: index + 1])!r}"
            )
        current = records[0]
    raise ValueError(f"empty condition {condition!r}")


def declared_condition_errors(
    condition: Condition, options_type: Any, *, root: bool = True
) -> list[str]:
    """Check every option path of ``condition`` against a declared options type.

    The generic ``OptionsConfig`` base declares nothing, so conditions of a
    model typed that loosely are checked only at construction. With
    ``root=False`` (``options_type`` is a nested record, not the root),
    atoms reading root-relative ``"/..."`` paths are left to the model class.
    """

    from hydroforge.contracts.options import OptionsConfig

    if not _is_record_type(options_type) or options_type is OptionsConfig:
        return []
    errors = []
    for atom in condition.option_atoms():
        if not root and any(is_root_path(path) for path in atom.paths()):
            continue
        try:
            atom.check_declared(options_type)
        except (TypeError, ValueError) as error:
            errors.append(str(error))
    return errors


def resolve_option_condition(options: OptionsConfig, path: str) -> bool:
    """Read an exact declared bool spelled ``options.<path>``."""

    if type(path) is not str or not path.startswith("options."):
        raise ValueError(f"not an options condition: {path!r}")
    return as_condition(path).evaluate(ConditionContext((), options=options))


def field_conditions(field: Any) -> tuple[Condition, ...]:
    """Return the ``depends_on`` and ``required_by`` conditions of one field."""

    tensor = field.tensor
    return (*tensor.depends_on, *tensor.required_by)


def resolve_conditions(
    modules: Iterable[Any],
    options: OptionsConfig | None,
    opened_modules: Iterable[str] | None = None,
) -> Mapping[str, bool]:
    """Resolve every option atom of opened owners' fields, including false ANDs.

    Keys are ``str(atom)``; an exact-bool flag keeps its historical
    ``"options.<path>"`` spelling. An atom that cannot be read (for example
    one traversing a ``None`` record) is left out when a guard such as
    ``opt("record").is_set() & ...`` short-circuits it for ``opened_modules``
    (default: the given modules); otherwise its error is raised. A path the
    options type does not declare is always an error.
    """

    specs = tuple(modules)
    values: dict[str, bool] = {}
    unreadable: set[str] = set()
    for module_spec in specs:
        for field in module_spec.tensor_fields.values():
            for condition in field_conditions(field):
                for atom in condition.option_atoms():
                    key = str(atom)
                    if key in values or key in unreadable:
                        continue
                    try:
                        if options is None:
                            raise ValueError("provide an explicit options context")
                        values[key] = atom.resolve(options)
                    except (TypeError, ValueError):
                        if options is not None:
                            # Only a declared path may be short-circuited.
                            try:
                                atom.check_declared(type(options))
                            except (TypeError, ValueError) as error:
                                raise ValueError(
                                    f"{module_spec.name}.{field.name}: {error}"
                                ) from error
                        unreadable.add(key)
    if unreadable:
        # Evaluate like ``tensor_is_active``: an unreadable atom that the
        # evaluation reaches re-raises its error, now naming the field.
        context = ConditionContext(
            frozenset(
                (item.name for item in specs)
                if opened_modules is None
                else opened_modules
            ),
            options=options,
            resolved=values,
        )
        for module_spec in specs:
            for field in module_spec.tensor_fields.values():
                tensor = field.tensor
                try:
                    if all(item.evaluate(context) for item in tensor.depends_on):
                        any(item.evaluate(context) for item in tensor.required_by)
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f"{module_spec.name}.{field.name}: {error}"
                    ) from error
    return MappingProxyType(values)


def conditions_satisfied(
    dependencies: Iterable[Any],
    opened_modules: Iterable[str],
    conditions: Mapping[str, bool] | None = None,
) -> bool:
    """Return whether every dependency (AND) holds for a resolved selection."""

    context = ConditionContext(
        frozenset(opened_modules),
        resolved=MappingProxyType({}) if conditions is None else conditions,
    )
    return all(as_condition(item).evaluate(context) for item in dependencies)


def any_condition_satisfied(
    consumers: Iterable[Any],
    opened_modules: Iterable[str],
    conditions: Mapping[str, bool] | None = None,
) -> bool:
    """Return whether at least one consumer condition (OR) holds."""

    context = ConditionContext(
        frozenset(opened_modules),
        resolved=MappingProxyType({}) if conditions is None else conditions,
    )
    return any(as_condition(item).evaluate(context) for item in consumers)


__all__ = [
    "AllOf",
    "AnyOf",
    "Condition",
    "ConditionContext",
    "Constant",
    "ModuleOpen",
    "Not",
    "OptionAtom",
    "OptionCompare",
    "OptionFlag",
    "OptionIn",
    "OptionIsSet",
    "OptionPairCompare",
    "OptionRef",
    "OptionTrait",
    "always",
    "any_condition_satisfied",
    "as_condition",
    "conditions_satisfied",
    "declared_condition_errors",
    "module",
    "never",
    "is_root_path",
    "opt",
    "prefix_path",
    "resolve_conditions",
    "resolve_option_condition",
]

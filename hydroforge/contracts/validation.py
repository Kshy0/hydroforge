"""Shared Pydantic base for declarative HydroForge API objects."""

from __future__ import annotations

from copy import deepcopy
from collections.abc import Mapping
from typing import Any, NoReturn

from pydantic import BaseModel, ConfigDict

_PUBLIC_MODEL_CONFIG = ConfigDict(
    arbitrary_types_allowed=True,
    extra="forbid",
    frozen=True,
    strict=True,
    validate_default=True,
    revalidate_instances="never",
)


class HydroForgeModel(BaseModel):
    """Strict immutable base for objects exposed by HydroForge's public API.

    Public fields are the object's complete declarative input.  Derived plans,
    caches and external resources belong in ``PrivateAttr`` attributes instead
    of a second Config or Contract model.
    """

    model_config = _PUBLIC_MODEL_CONFIG

    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> NoReturn:
        """Reject copies that would alias or duplicate private runtime state."""

        del update, deep
        raise TypeError(
            "HydroForge identities cannot be copied with model_copy(); "
            "construct a new validated object or use the object's validated "
            "functional update API"
        )

    def copy(self, *args: Any, **kwargs: Any) -> NoReturn:
        del args, kwargs
        raise TypeError(
            "Pydantic copy(include/exclude/update=...) bypasses validation for "
            "frozen HydroForge models; construct a new validated object or "
            "use a validated functional update API"
        )

    def __copy__(self) -> NoReturn:
        raise TypeError(
            "HydroForge identities cannot be copied with copy.copy(); "
            "construct a new validated object"
        )

    def __deepcopy__(self, memo: dict[int, Any] | None = None) -> NoReturn:
        del memo
        raise TypeError(
            "HydroForge identities cannot be copied with copy.deepcopy(); "
            "construct a new validated object"
        )


class _ImmutableDict(dict[Any, Any]):
    """A serializer-friendly immutable mapping for frozen model fields."""

    __slots__ = ()

    @staticmethod
    def _reject_mutation(*args: Any, **kwargs: Any) -> None:
        del args, kwargs
        raise TypeError("frozen HydroForge model mappings are immutable")

    __setitem__ = _reject_mutation
    __delitem__ = _reject_mutation
    __ior__ = _reject_mutation
    clear = _reject_mutation
    pop = _reject_mutation
    popitem = _reject_mutation
    setdefault = _reject_mutation
    update = _reject_mutation

    def __copy__(self) -> _ImmutableDict:
        return self

    def __deepcopy__(self, memo: dict[int, Any]) -> _ImmutableDict:
        copied = type(self)(
            (deepcopy(key, memo), deepcopy(value, memo))
            for key, value in self.items()
        )
        memo[id(self)] = copied
        return copied

    def copy(self) -> _ImmutableDict:
        return self

    def __reduce__(self):
        # The default dict-subclass pickle protocol constructs an empty object
        # and then restores entries through ``__setitem__``.  That is correctly
        # forbidden for this immutable public mapping, so reconstruct it in one
        # constructor call instead.  This keeps DataLoader spawn/forkserver
        # workers serializable without opening a mutation backdoor.
        return (_restore_immutable_dict, (tuple(self.items()),))

    def __reduce_ex__(self, protocol: int):
        del protocol
        return self.__reduce__()


def _restore_immutable_dict(items: tuple[tuple[Any, Any], ...]) -> _ImmutableDict:
    return _ImmutableDict(items)


def _immutable_dict(values: Any) -> _ImmutableDict:
    return _ImmutableDict(dict(values))


__all__ = ["HydroForgeModel"]

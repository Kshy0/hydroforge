# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""Compiled field names bound to the constructed module instances."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from hydroforge.compiler.fields import BindingSource, FieldEntry


@dataclass(frozen=True, slots=True)
class NamespaceEntry:
    """Resolved owner and coordinate metadata for a model field."""

    module: Any
    field_name: str
    coordinate: str | None


@dataclass(frozen=True, slots=True)
class FieldOwner:
    """One owner a kernel ABI name binds to."""

    module_name: str
    field_name: str
    owner: Any


def bind_namespace(
    names: Mapping[str, FieldEntry], modules: Mapping[str, Any]
) -> Mapping[str, NamespaceEntry]:
    """Bind qualified and unambiguous bare field names to module instances."""

    return MappingProxyType(
        {
            name: NamespaceEntry(
                modules[entry.module], entry.name, entry.tensor.dim_coords
            )
            for name, entry in names.items()
        }
    )


def bind_field_owners(
    sources: Mapping[str, tuple[BindingSource, ...]],
    modules: Mapping[str, Any],
    model: Any,
) -> Mapping[str, tuple[FieldOwner, ...]]:
    """Bind kernel ABI names to the modules, model and model records owning them.

    Buffer virtuals bind only once initialization has materialized them, so
    index construction never turns a declared diagnostic into resident state.
    """

    index: dict[str, tuple[FieldOwner, ...]] = {}
    for name, candidates in sources.items():
        owners = []
        for source in candidates:
            if source.owner == "model":
                owner = model
            elif source.owner.startswith("model."):
                owner = getattr(model, source.owner.removeprefix("model."))
            else:
                owner = modules[source.owner]
            if not source.lazy or name in owner.__dict__:
                owners.append(FieldOwner(source.owner, name, owner))
        if owners:
            index[name] = tuple(owners)
    return MappingProxyType(index)

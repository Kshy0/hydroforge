"""Model namespace resolution isolated from the public model API."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Generic, TypeVar

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel


@dataclass(frozen=True, slots=True)
class NamespaceEntry:
    """Resolved owner and coordinate metadata for a model field."""

    module: object
    field_name: str
    coordinate: str | None


_Entry = TypeVar("_Entry")


class FieldNameResolver(Generic[_Entry]):
    """One qualified/bare-name precedence rule for declarations and bindings."""

    def __init__(self) -> None:
        self.entries: dict[str, _Entry] = {}
        self.virtual_names: set[str] = set()
        self.ambiguous_names: set[str] = set()

    def install(
        self,
        module_name: str,
        field_name: str,
        entry: _Entry,
        *,
        expression_virtual: bool,
    ) -> None:
        self.entries[f"{module_name}.{field_name}"] = entry
        if expression_virtual:
            if field_name in self.virtual_names:
                self.entries.pop(field_name, None)
                self.ambiguous_names.add(field_name)
            else:
                self.entries[field_name] = entry
                self.virtual_names.add(field_name)
                self.ambiguous_names.discard(field_name)
        elif (
            field_name not in self.virtual_names
            and field_name not in self.ambiguous_names
        ):
            if field_name in self.entries:
                self.entries.pop(field_name)
                self.ambiguous_names.add(field_name)
            else:
                self.entries[field_name] = entry


class NamespaceCompiler:
    """Build qualified mappings and resolve unqualified field ownership."""

    def __init__(self, model: AbstractModel) -> None:
        self.model = model
        self._mapping: Mapping[str, NamespaceEntry] | None = None

    def build(self) -> Mapping[str, NamespaceEntry]:
        if self._mapping is not None:
            return self._mapping
        resolver: FieldNameResolver[NamespaceEntry] = FieldNameResolver()

        for module_name in self.model.opened_modules:
            module = self.model._modules[module_name]
            for field in module.tensor_schema():
                if not module._is_tensor_field_active(field):
                    continue
                field_name = field.name
                entry = NamespaceEntry(
                    module=module,
                    field_name=field_name,
                    coordinate=field.tensor.dim_coords,
                )
                resolver.install(
                    module_name,
                    field_name,
                    entry,
                    expression_virtual=(
                        field.tensor.category == "virtual"
                        and bool(field.tensor.expression)
                    ),
                )

            for field_name in module._reference_index_fields(
                opened_modules=self.model.opened_modules,
                field_demand=self.model._field_demand,
            ):
                metadata = module._reference_index_metadata(
                    field_name,
                    opened_modules=self.model.opened_modules,
                    field_demand=self.model._field_demand,
                )
                entry = NamespaceEntry(
                    module=module,
                    field_name=field_name,
                    coordinate=metadata.dim_coords,
                )
                resolver.install(
                    module_name,
                    field_name,
                    entry,
                    expression_virtual=False,
                )
        self._mapping = MappingProxyType(resolver.entries)
        return self._mapping

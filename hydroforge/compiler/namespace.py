"""Model namespace resolution isolated from the public model API."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from hydroforge.contracts.fields import tensor_is_active

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel
    from hydroforge.model.module import AbstractModule


@dataclass(frozen=True, slots=True)
class NamespaceEntry:
    """Resolved owner and coordinate metadata for a model field."""

    module: AbstractModule
    field_name: str
    coordinate: str | None


class NamespaceCompiler:
    """Build qualified mappings and reject ambiguous unqualified fields."""

    def __init__(self, model: AbstractModel) -> None:
        self.model = model
        self._mapping: Mapping[str, NamespaceEntry] | None = None

    def build(self) -> Mapping[str, NamespaceEntry]:
        if self._mapping is not None:
            return self._mapping
        mapping: dict[str, NamespaceEntry] = {}
        virtual: dict[str, bool] = {}
        ambiguous: set[str] = set()
        for module_name in self.model.opened_modules:
            module = self.model._modules[module_name]
            for field in module.tensor_schema():
                if not tensor_is_active(field.tensor, self.model.opened_modules):
                    continue
                field_name = field.name
                entry = NamespaceEntry(
                    module=module,
                    field_name=field_name,
                    coordinate=field.tensor.dim_coords,
                )
                is_virtual = (
                    field.tensor.category == "virtual"
                    and bool(field.tensor.expression)
                )
                if field_name in ambiguous:
                    pass
                elif field_name not in mapping:
                    mapping[field_name] = entry
                    virtual[field_name] = is_virtual
                elif is_virtual and not virtual.get(field_name):
                    mapping[field_name] = entry
                    virtual[field_name] = True
                elif not is_virtual and virtual.get(field_name):
                    pass
                else:
                    mapping.pop(field_name, None)
                    ambiguous.add(field_name)
                mapping[f"{module_name}.{field_name}"] = entry

            for field_name in module.get_reference_index_fields():
                metadata = module.get_reference_index_metadata(field_name)
                entry = NamespaceEntry(
                    module=module,
                    field_name=field_name,
                    coordinate=metadata.dim_coords,
                )
                if field_name not in ambiguous:
                    if field_name in mapping and mapping[field_name] != entry:
                        mapping.pop(field_name)
                        ambiguous.add(field_name)
                    else:
                        mapping.setdefault(field_name, entry)
                mapping[f"{module_name}.{field_name}"] = entry
        self._mapping = MappingProxyType(mapping)
        return self._mapping

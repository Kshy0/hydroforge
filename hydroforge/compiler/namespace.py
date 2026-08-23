"""Model namespace resolution isolated from the public model API."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from hydroforge.contracts.fields import tensor_is_active

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel


@dataclass(frozen=True, slots=True)
class NamespaceEntry:
    """Resolved owner and coordinate metadata for a model field."""

    module: object
    field_name: str
    coordinate: str | None


class NamespaceCompiler:
    """Build qualified mappings and resolve unqualified field ownership."""

    def __init__(self, model: AbstractModel) -> None:
        self.model = model
        self._mapping: Mapping[str, NamespaceEntry] | None = None

    def build(self) -> Mapping[str, NamespaceEntry]:
        if self._mapping is not None:
            return self._mapping
        mapping: dict[str, NamespaceEntry] = {}
        virtual: set[str] = set()
        ambiguous: set[str] = set()

        def install_bare(
            field_name: str,
            entry: NamespaceEntry,
            *,
            expression_virtual: bool,
        ) -> None:
            if expression_virtual:
                if field_name not in virtual:
                    mapping[field_name] = entry
                    virtual.add(field_name)
                ambiguous.discard(field_name)
                return
            if field_name in virtual or field_name in ambiguous:
                return
            if field_name in mapping:
                mapping.pop(field_name)
                ambiguous.add(field_name)
                return
            mapping[field_name] = entry

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
                mapping[f"{module_name}.{field_name}"] = entry
                install_bare(
                    field_name,
                    entry,
                    expression_virtual=(
                        field.tensor.category == "virtual"
                        and bool(field.tensor.expression)
                    ),
                )

            for field_name in module._reference_index_fields():
                metadata = module._reference_index_metadata(field_name)
                entry = NamespaceEntry(
                    module=module,
                    field_name=field_name,
                    coordinate=metadata.dim_coords,
                )
                mapping[f"{module_name}.{field_name}"] = entry
                install_bare(
                    field_name,
                    entry,
                    expression_virtual=False,
                )
        self._mapping = MappingProxyType(mapping)
        return self._mapping

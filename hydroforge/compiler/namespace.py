"""Model namespace resolution isolated from the public model API."""

from __future__ import annotations

from typing import Any


class NamespaceCompiler:
    """Build qualified mappings and reject ambiguous unqualified fields."""

    def __init__(self, model: Any) -> None:
        self.model = model
        self._mapping: dict[str, tuple[Any, str, str | None]] | None = None

    def build(self) -> dict[str, tuple[Any, str, str | None]]:
        if self._mapping is not None:
            return self._mapping
        mapping: dict[str, tuple[Any, str, str | None]] = {}
        virtual: dict[str, bool] = {}
        ambiguous: set[str] = set()
        for module_name in self.model.opened_modules:
            module = self.model.get_module(module_name)
            if module is None:
                raise RuntimeError(
                    f"opened module {module_name!r} is absent from the compiled "
                    "model namespace"
                )
            for field in module.tensor_schema():
                field_name = field.name
                entry = (module, field_name, field.tensor.dim_coords)
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
                entry = (module, field_name, metadata.dim_coords)
                if field_name not in ambiguous:
                    if field_name in mapping and mapping[field_name] != entry:
                        mapping.pop(field_name)
                        ambiguous.add(field_name)
                    else:
                        mapping.setdefault(field_name, entry)
                mapping[f"{module_name}.{field_name}"] = entry
        self._mapping = mapping
        return mapping

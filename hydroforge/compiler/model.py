"""Cold-path field namespace compiler for one model specialization."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import TYPE_CHECKING, Mapping

from pydantic import BaseModel

from hydroforge.contracts.kernel_field import KernelField
from hydroforge.contracts.fields import tensor_is_active

if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel


@dataclass(frozen=True, slots=True)
class FieldOwner:
    module_name: str
    field_name: str
    owner: BaseModel


class FieldNamespaceCompiler:
    """Compile declarative model fields into an immutable lookup index."""

    def __init__(self, model: AbstractModel) -> None:
        self.model = model

    def compile(self) -> Mapping[str, tuple[FieldOwner, ...]]:
        return MappingProxyType(self._field_owners())

    def _field_owners(self) -> dict[str, tuple[FieldOwner, ...]]:
        model = self.model
        index: dict[str, list[FieldOwner]] = {}
        for module_name, module in model._modules.items():
            fields = {
                name
                for name, schema in module.field_schema_map().items()
                if (
                    schema.tensor is None
                    or (
                        not schema.tensor.expression
                        and tensor_is_active(
                            schema.tensor, model.opened_modules,
                        )
                        and (
                            schema.tensor.category != "virtual"
                            or name in module.__dict__
                        )
                    )
                )
            } | set(module.get_reference_index_fields())
            for field_name in fields:
                index.setdefault(field_name, []).append(FieldOwner(
                    module_name=module_name,
                    field_name=field_name,
                    owner=module,
                ))
            for cls in reversed(type(module).__mro__):
                for field_name, descriptor in vars(cls).items():
                    if not isinstance(descriptor, KernelField):
                        continue
                    index.setdefault(field_name, []).append(FieldOwner(
                        module_name=module_name,
                        field_name=field_name,
                        owner=module,
                    ))
        for field_name in model.__class__.model_fields:
            index.setdefault(field_name, []).append(FieldOwner(
                module_name="model",
                field_name=field_name,
                owner=model,
            ))
            value = getattr(model, field_name)
            if (
                isinstance(value, BaseModel)
                and type(value).model_config.get("frozen") is True
            ):
                for nested_name in type(value).model_fields:
                    index.setdefault(nested_name, []).append(FieldOwner(
                        module_name=f"model.{field_name}",
                        field_name=nested_name,
                        owner=value,
                    ))
        for cls in reversed(type(model).__mro__):
            for field_name, descriptor in vars(cls).items():
                if not isinstance(descriptor, KernelField):
                    continue
                index.setdefault(field_name, []).append(FieldOwner(
                    module_name="model",
                    field_name=field_name,
                    owner=model,
                ))
        return {
            field_name: tuple(owners)
            for field_name, owners in index.items()
        }

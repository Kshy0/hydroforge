"""Between-step structural tensor updates derived from declared dimensions."""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
import inspect
from numbers import Integral
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Callable, Iterable, Mapping

import torch


if TYPE_CHECKING:
    from hydroforge.model.model import AbstractModel


@dataclass(frozen=True, slots=True)
class StructuralUpdateResult:
    """Committed model-structure revision and inferred dimension changes."""

    revision: int
    dimensions: Mapping[str, tuple[int, int]]
    invalidated: bool = True


class StructuralUpdateContext:
    """One ordered module pass that stages a single structural transaction."""

    def __init__(
        self,
        model: AbstractModel,
    ) -> None:
        self._model = model
        self._bindings: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._content_bindings: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._target_ids: set[int] = set()
        self._finalizers: list[
            Callable[[StructuralUpdateResult], None]
        ] = []

    def require_output_coordinate_resize_safe(
        self,
        coordinate: str,
        *,
        old_extent: int,
        new_extent: int,
        stable_scatter_index: str | None = None,
        stable_output_coordinate: str | None = None,
    ) -> None:
        """Reject growth that would resize an installed output domain."""

        statistics = getattr(self._model._statistics, "aggregator", None)
        if statistics is not None:
            statistics.require_output_coordinate_resize_safe(
                coordinate,
                old_extent=old_extent,
                new_extent=new_extent,
                stable_scatter_index=stable_scatter_index,
                stable_output_coordinate=stable_output_coordinate,
            )

    def stage(
        self,
        bindings: Iterable[tuple[torch.Tensor, torch.Tensor]],
        *,
        after_commit: Callable[[StructuralUpdateResult], None] | None = None,
    ) -> None:
        """Stage one module's replacements without mutating live state."""

        pairs = tuple(bindings)
        if not pairs:
            raise ValueError("a module structural update cannot be empty")
        identities = [id(current) for current, _replacement in pairs]
        duplicates = self._target_ids.intersection(identities)
        if duplicates or len(set(identities)) != len(identities):
            raise ValueError(
                "module structural updates contain duplicate target tensors"
            )
        if after_commit is not None and not callable(after_commit):
            raise TypeError("structural after_commit callback must be callable")
        self._bindings.extend(pairs)
        self._target_ids.update(identities)
        if after_commit is not None:
            self._finalizers.append(after_commit)

    def stage_content(
        self,
        bindings: Iterable[tuple[torch.Tensor, torch.Tensor]],
        *,
        after_commit: Callable[[StructuralUpdateResult], None] | None = None,
    ) -> None:
        """Stage same-shape content changes that preserve captured addresses."""

        pairs = tuple(bindings)
        if not pairs:
            raise ValueError("a module content update cannot be empty")
        identities = [id(current) for current, _replacement in pairs]
        duplicates = self._target_ids.intersection(identities)
        if duplicates or len(set(identities)) != len(identities):
            raise ValueError(
                "module updates contain duplicate target tensors"
            )
        if after_commit is not None and not callable(after_commit):
            raise TypeError("content after_commit callback must be callable")
        self._content_bindings.extend(pairs)
        self._target_ids.update(identities)
        if after_commit is not None:
            self._finalizers.append(after_commit)

    def commit(self) -> StructuralUpdateResult | None:
        """Commit all staged modules once, then finalize them in call order."""

        if not self._bindings and not self._content_bindings:
            return None
        if self._bindings:
            result = commit_structural_update(
                self._model,
                (*self._bindings, *self._content_bindings),
            )
        else:
            result = commit_content_update(
                self._model,
                self._content_bindings,
            )
        try:
            for finalize in self._finalizers:
                finalize(result)
        except BaseException as error:
            self._model._execution.poison(
                error,
                phase="module structural update finalization",
            )
            raise
        return result


@dataclass(frozen=True, slots=True)
class _Dimension:
    owner: Any
    attribute: str
    label: str

    @property
    def key(self) -> tuple[int, str]:
        return id(self.owner), self.attribute


def _dimension(module: Any, token: str) -> _Dimension:
    if "." in token:
        owner_name, attribute = token.split(".", 1)
        owner = getattr(module, owner_name, None)
        if owner is None:
            raise ValueError(
                f"dimension {token!r} has no owner in module "
                f"{module.module_name!r}"
            )
        label = f"{owner.module_name}.{attribute}"
    else:
        owner = module
        attribute = token
        label = f"{module.module_name}.{attribute}"
    if not hasattr(owner, attribute):
        raise ValueError(f"dimension {label!r} is not materialized")
    return _Dimension(owner, attribute, label)


def _logical_shape(
    module: Any,
    field_name: str,
    tensor: torch.Tensor,
    rank: int,
) -> tuple[int, ...]:
    current = getattr(module, field_name)
    batched = (
        isinstance(current, torch.Tensor)
        and module._is_batched_trusted(field_name)
    )
    expected_rank = rank + int(batched)
    if tensor.ndim != expected_rank:
        raise ValueError(
            f"structural replacement {module.module_name}.{field_name} has "
            f"rank {tensor.ndim}, expected {expected_rank}"
        )
    if batched:
        if tensor.shape[0] != module.num_trials:
            raise ValueError(
                f"structural replacement {module.module_name}.{field_name} "
                f"has trial extent {tensor.shape[0]}, expected "
                f"{module.num_trials}"
            )
        return tuple(tensor.shape[1:])
    return tuple(tensor.shape)


def _replacement_fields(
    model: AbstractModel,
    replacements: Mapping[int, torch.Tensor],
) -> dict[int, list[tuple[Any, str, Any]]]:
    fields: dict[int, list[tuple[Any, str, Any]]] = {
        identity: [] for identity in replacements
    }
    for field_name, owners in model._field_namespace.items():
        for entry in owners:
            value = getattr(entry.owner, field_name)
            identity = id(value)
            if identity not in fields:
                continue
            schema_getter = getattr(entry.owner, "_get_tensor_schema", None)
            schema = (
                None if schema_getter is None else schema_getter(field_name)
            )
            fields[identity].append((entry.owner, field_name, schema))
    missing = [identity for identity, matches in fields.items() if not matches]
    if missing:
        raise ValueError(
            "structural replacements must target declared model tensors; "
            f"unowned tensor identities={missing}"
        )
    return fields


def _infer_dimensions(
    fields: Mapping[int, list[tuple[Any, str, Any]]],
    replacements: Mapping[int, torch.Tensor],
) -> dict[tuple[int, str], tuple[_Dimension, int]]:
    inferred: dict[tuple[int, str], tuple[_Dimension, int]] = {}
    for identity, matches in fields.items():
        replacement = replacements[identity]
        for module, field_name, schema in matches:
            if schema is None or schema.tensor is None:
                continue
            declared = schema.tensor.shape
            actual = _logical_shape(
                module, field_name, replacement, len(declared),
            )
            for token, extent in zip(declared, actual, strict=True):
                if isinstance(token, int):
                    if extent != token:
                        raise ValueError(
                            f"structural replacement "
                            f"{module.module_name}.{field_name} dimension "
                            f"must remain {token}, got {extent}"
                        )
                    continue
                dimension = _dimension(module, token)
                prior = inferred.get(dimension.key)
                if prior is not None and prior[1] != extent:
                    raise ValueError(
                        f"structural replacements infer conflicting extents "
                        f"for {dimension.label}: {prior[1]} and {extent}"
                    )
                inferred[dimension.key] = dimension, extent
    return inferred


def _expected_shape(
    module: Any,
    field_name: str,
    declared: tuple[Any, ...],
    inferred: Mapping[tuple[int, str], tuple[_Dimension, int]],
) -> tuple[int, ...]:
    dimensions: list[int] = []
    for token in declared:
        if isinstance(token, int):
            dimensions.append(token)
            continue
        dimension = _dimension(module, token)
        inferred_value = inferred.get(dimension.key)
        value = (
            inferred_value[1]
            if inferred_value is not None
            else getattr(dimension.owner, dimension.attribute)
        )
        if isinstance(value, bool) or not isinstance(value, Integral):
            raise ValueError(
                f"dimension {dimension.label!r} must be an integer, got "
                f"{type(value).__name__}"
            )
        dimensions.append(int(value))
    current = getattr(module, field_name)
    if (
        isinstance(current, torch.Tensor)
        and module._is_batched_trusted(field_name)
    ):
        return (module.num_trials, *dimensions)
    return tuple(dimensions)


def _validate_dependent_shapes(
    model: AbstractModel,
    replacements: Mapping[int, torch.Tensor],
    inferred: Mapping[tuple[int, str], tuple[_Dimension, int]],
) -> None:
    changed = sorted(
        dimension.label
        for dimension, new_value in inferred.values()
        if int(getattr(dimension.owner, dimension.attribute)) != new_value
    )
    for module_name in model.opened_modules:
        module = model._modules[module_name]
        for field in module.tensor_schema():
            tensor_schema = field.tensor
            if not module._is_tensor_field_active(field):
                continue
            if (
                tensor_schema.category == "virtual"
                and field.name not in module.__dict__
            ):
                continue
            current = getattr(module, field.name, None)
            if not isinstance(current, torch.Tensor):
                continue
            candidate = replacements.get(id(current), current)
            expected = _expected_shape(
                module, field.name, tensor_schema.shape, inferred,
            )
            if tuple(candidate.shape) != expected:
                raise ValueError(
                    f"structural update leaves {module_name}.{field.name} at "
                    f"shape {tuple(candidate.shape)}, expected {expected}; "
                    f"changed dimensions={changed}"
                )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "xpu":
        runtime = getattr(torch, "xpu", None)
        if runtime is not None:
            runtime.synchronize(device)


def _publish_dimensions(
    inferred: Mapping[tuple[int, str], tuple[_Dimension, int]],
    previous: Mapping[tuple[int, str], int],
) -> dict[str, tuple[int, int]]:
    changes: dict[str, tuple[int, int]] = {}
    for key, (dimension, extent) in inferred.items():
        old = previous[key]
        if old == extent:
            continue
        descriptor = inspect.getattr_static(
            type(dimension.owner), dimension.attribute, None,
        )
        if isinstance(descriptor, cached_property):
            dimension.owner.__dict__.pop(dimension.attribute, None)
        elif not isinstance(descriptor, property):
            object.__setattr__(dimension.owner, dimension.attribute, extent)
        changes[dimension.label] = (old, extent)
    return changes


def _verify_dimensions(
    inferred: Mapping[tuple[int, str], tuple[_Dimension, int]],
) -> None:
    for dimension, extent in inferred.values():
        observed = getattr(dimension.owner, dimension.attribute)
        if observed != extent:
            raise RuntimeError(
                f"dimension {dimension.label!r} resolved to {observed}, "
                f"expected {extent} after structural update"
            )


def commit_content_update(
    model: AbstractModel,
    bindings: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> StructuralUpdateResult:
    """Copy same-shape declared tensors without invalidating captured execution."""

    from hydroforge.execution.step import _managed_step_active

    if _managed_step_active():
        raise RuntimeError("content updates are allowed only between managed steps")
    pairs = tuple(bindings)
    if not pairs:
        raise ValueError("content update requires at least one replacement")
    current_ids = [id(current) for current, _replacement in pairs]
    if len(set(current_ids)) != len(current_ids):
        raise ValueError("content update contains duplicate target tensors")
    replacements: dict[int, torch.Tensor] = {}
    for current, replacement in pairs:
        if not isinstance(current, torch.Tensor) or not isinstance(
            replacement, torch.Tensor,
        ):
            raise TypeError("content replacements must be tensor pairs")
        if current is replacement:
            raise ValueError("content replacement must use staged storage")
        if current.dtype != replacement.dtype:
            raise TypeError(
                f"content replacement changes dtype from {current.dtype} "
                f"to {replacement.dtype}"
            )
        if current.device != replacement.device:
            raise ValueError(
                f"content replacement changes device from {current.device} "
                f"to {replacement.device}"
            )
        if current.shape != replacement.shape:
            raise ValueError(
                f"content replacement changes shape from {tuple(current.shape)} "
                f"to {tuple(replacement.shape)}"
            )
        if replacement.layout is not torch.strided or not replacement.is_contiguous():
            raise ValueError(
                "content replacement must be a contiguous strided tensor"
            )
        replacements[id(current)] = replacement

    fields = _replacement_fields(model, replacements)
    for identity, matches in fields.items():
        for _module, field_name, schema in matches:
            if (
                schema is not None
                and schema.tensor is not None
                and schema.tensor.is_coordinate
                and not torch.equal(
                    next(current for current, _ in pairs if id(current) == identity),
                    replacements[identity],
                )
            ):
                raise ValueError(
                    f"address-stable content update cannot change coordinate "
                    f"{field_name!r}"
                )

    mutated = False
    try:
        _synchronize(torch.device(model.device))
        mutated = True
        with torch.inference_mode():
            for current, replacement in pairs:
                current.copy_(replacement)
        statistics = getattr(model._statistics, "aggregator", None)
        if statistics is not None:
            statistics.refresh_address_stable_sources()
        return StructuralUpdateResult(
            revision=model._execution.structural_revision,
            dimensions=MappingProxyType({}),
            invalidated=False,
        )
    except BaseException as error:
        if mutated:
            model._execution.poison(error, phase="address-stable content update")
        raise


def commit_structural_update(
    model: AbstractModel,
    bindings: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> StructuralUpdateResult:
    """Rebind declared tensors and rebuild every structural consumer.

    Changed symbolic dimensions are inferred from tensor field schemas, so a
    model implementation only stages replacement storage. Every field that
    depends on an inferred dimension must be supplied at its new shape;
    otherwise validation fails before cached execution is invalidated.
    """

    from hydroforge.execution.step import _managed_step_active

    if _managed_step_active():
        raise RuntimeError(
            "structural updates are allowed only between managed steps"
        )
    pairs = tuple(bindings)
    if not pairs:
        raise ValueError("structural update requires at least one replacement")
    current_ids = [id(current) for current, _replacement in pairs]
    if len(set(current_ids)) != len(current_ids):
        raise ValueError("structural update contains duplicate target tensors")
    replacements: dict[int, torch.Tensor] = {}
    for current, replacement in pairs:
        if not isinstance(current, torch.Tensor) or not isinstance(
            replacement, torch.Tensor,
        ):
            raise TypeError("structural replacements must be tensor pairs")
        if current is replacement:
            raise ValueError("structural replacement must use new storage")
        if current.dtype != replacement.dtype:
            raise TypeError(
                f"structural replacement changes dtype from {current.dtype} "
                f"to {replacement.dtype}"
            )
        if current.device != replacement.device:
            raise ValueError(
                f"structural replacement changes device from "
                f"{current.device} to {replacement.device}"
            )
        if (
            replacement.layout is not torch.strided
            or not replacement.is_contiguous()
        ):
            raise ValueError(
                "structural replacement must be a contiguous strided tensor"
            )
        replacements[id(current)] = replacement

    fields = _replacement_fields(model, replacements)
    inferred = _infer_dimensions(fields, replacements)
    _validate_dependent_shapes(model, replacements, inferred)
    previous_dimensions = {
        key: int(getattr(dimension.owner, dimension.attribute))
        for key, (dimension, _extent) in inferred.items()
    }

    mutated = False
    try:
        _synchronize(torch.device(model.device))
        model._execution.invalidate()
        mutated = True
        with torch.inference_mode():
            for current, replacement in pairs:
                current.set_(replacement)
        changes = _publish_dimensions(inferred, previous_dimensions)
        _verify_dimensions(inferred)

        statistics = getattr(model._statistics, "aggregator", None)
        if statistics is not None:
            statistics.recompile_resized_sources()

        from hydroforge.output.checkpoint import CheckpointRuntime

        model._checkpoint = CheckpointRuntime(model)
        model._execution.structural_revision += 1
        return StructuralUpdateResult(
            revision=model._execution.structural_revision,
            dimensions=MappingProxyType(changes),
        )
    except BaseException as error:
        if mutated:
            model._execution.poison(error, phase="structural tensor update")
        raise

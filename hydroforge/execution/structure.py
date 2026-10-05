"""Between-step structural tensor updates derived from declared dimensions."""

from __future__ import annotations

import inspect
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from numbers import Integral
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Literal

import torch

from hydroforge.core.arrays import find_indices_in_torch
from hydroforge.declare.tensors import ModuleTensors
from hydroforge.execution.checkpoint import CheckpointRuntime
from hydroforge.execution.context import managed_step_active, validate_callback

if TYPE_CHECKING:
    from hydroforge.execution.session import ModelRuntime


@dataclass(frozen=True, slots=True)
class StructuralUpdateResult:
    """Committed model-structure revision and inferred dimension changes."""

    revision: int
    dimensions: Mapping[str, tuple[int, int]]
    invalidated: bool = True


def _tensor_pairs(
    bindings: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Collect staged ``(current, replacement)`` pairs with distinct targets."""

    try:
        pairs = tuple(tuple(pair) for pair in bindings)
    except TypeError as error:
        raise ValueError(
            "tensor update bindings must be iterable tensor pairs"
        ) from error
    if not pairs:
        raise ValueError("tensor update bindings must not be empty")
    for pair in pairs:
        if len(pair) != 2 or not all(isinstance(item, torch.Tensor) for item in pair):
            raise TypeError(
                "tensor update bindings must be (current, replacement) tensor pairs"
            )
    if len({id(current) for current, _ in pairs}) != len(pairs):
        raise ValueError("tensor update contains duplicate target tensors")
    return pairs


class StructuralUpdateContext:
    """One ordered module pass that stages a single structural transaction."""

    def __init__(self, runtime: ModelRuntime) -> None:
        self._runtime = runtime
        self._bindings: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._content_bindings: list[tuple[torch.Tensor, torch.Tensor]] = []
        self._target_ids: set[int] = set()
        self._finalizers: list[Callable[[StructuralUpdateResult], None]] = []

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

        statistics = self._runtime.statistics
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

        self._stage(bindings, after_commit=after_commit, target=self._bindings)

    def stage_content(
        self,
        bindings: Iterable[tuple[torch.Tensor, torch.Tensor]],
        *,
        after_commit: Callable[[StructuralUpdateResult], None] | None = None,
    ) -> None:
        """Stage same-shape content changes that preserve captured addresses."""

        self._stage(bindings, after_commit=after_commit, target=self._content_bindings)

    def _stage(
        self,
        bindings: Iterable[tuple[torch.Tensor, torch.Tensor]],
        *,
        after_commit: Callable[[StructuralUpdateResult], None] | None,
        target: list[tuple[torch.Tensor, torch.Tensor]],
    ) -> None:
        pairs = _tensor_pairs(bindings)
        validate_callback(after_commit, arguments=1, label="after_commit")
        targets = {id(current) for current, _ in pairs}
        if self._target_ids.intersection(targets):
            raise ValueError("module updates contain duplicate target tensors")
        target.extend(pairs)
        self._target_ids.update(targets)
        if after_commit is not None:
            self._finalizers.append(after_commit)

    def commit(self) -> StructuralUpdateResult | None:
        """Commit all staged modules once, then finalize them in call order."""

        if not self._bindings and not self._content_bindings:
            return None
        if self._bindings:
            result = _commit_structural_update(
                self._runtime,
                (*self._bindings, *self._content_bindings),
                content_ids=frozenset(
                    id(current) for current, _ in self._content_bindings
                ),
            )
        else:
            result = _commit_content_update(
                self._runtime,
                self._content_bindings,
            )
        try:
            for finalize in self._finalizers:
                finalize(result)
        except BaseException as error:
            self._runtime.execution.poison(
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
                f"dimension {token!r} has no owner in module {module.module_name!r}"
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
    batched = isinstance(current, torch.Tensor) and module.is_batched(field_name)
    expected_rank = rank + int(batched)
    if tensor.ndim != expected_rank:
        raise ValueError(
            f"structural replacement {module.module_name}.{field_name} has "
            f"rank {tensor.ndim}, expected {expected_rank}"
        )
    if batched:
        if tensor.shape[0] != module.ensemble_size:
            raise ValueError(
                f"structural replacement {module.module_name}.{field_name} "
                f"has member extent {tensor.shape[0]}, expected "
                f"{module.ensemble_size}"
            )
        return tuple(tensor.shape[1:])
    return tuple(tensor.shape)


def _replacement_fields(
    runtime: ModelRuntime,
    replacements: Mapping[int, torch.Tensor],
) -> dict[int, list[tuple[Any, str, Any]]]:
    fields: dict[int, list[tuple[Any, str, Any]]] = {
        identity: [] for identity in replacements
    }
    for field_name, owners in runtime.field_owners.items():
        for entry in owners:
            value = getattr(entry.owner, field_name)
            identity = id(value)
            if identity not in fields:
                continue
            metadata_getter = getattr(entry.owner, "_tensor_metadata", None)
            metadata = None if metadata_getter is None else metadata_getter(field_name)
            fields[identity].append((entry.owner, field_name, metadata))
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
        for module, field_name, metadata in matches:
            if metadata is None:
                continue
            declared = metadata.shape
            actual = _logical_shape(
                module,
                field_name,
                replacement,
                len(declared),
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
    if isinstance(current, torch.Tensor) and module.is_batched(field_name):
        return (module.ensemble_size, *dimensions)
    return tuple(dimensions)


def _validate_dependent_shapes(
    runtime: ModelRuntime,
    replacements: Mapping[int, torch.Tensor],
    inferred: Mapping[tuple[int, str], tuple[_Dimension, int]],
) -> None:
    changed = sorted(
        dimension.label
        for dimension, new_value in inferred.values()
        if int(getattr(dimension.owner, dimension.attribute)) != new_value
    )
    for module_name in runtime.plan.modules:
        module = runtime.modules[module_name]
        for field in module.spec().tensor_fields.values():
            tensor_schema = field.tensor
            if not module._is_tensor_field_active(field.name):
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
                module,
                field.name,
                tensor_schema.shape,
                inferred,
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
            type(dimension.owner),
            dimension.attribute,
            None,
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


def _require_update_boundary(runtime: ModelRuntime, kind: str) -> None:
    """Check existing services without recursively materializing the runtime."""
    if managed_step_active():
        raise RuntimeError(f"{kind} updates are allowed only between managed steps")
    execution = runtime.execution
    if execution is None or execution.closed:
        raise RuntimeError(f"{kind} updates require an initialized, non-closed runtime")
    failure = execution.failure
    if failure is not None:
        raise execution.poisoned_error(failure)


def _validate_replacements(
    pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    kind: Literal["content", "structural"],
    content_ids: frozenset[int] = frozenset(),
) -> dict[int, torch.Tensor]:
    """Check live storage at commit; staged tensors may have changed since validation."""
    targets = set()
    for current, _ in pairs:
        if current.layout != torch.strided or not current.is_contiguous():
            raise ValueError(
                "tensor update targets must use contiguous strided storage"
            )
        identity = (current.device, current.untyped_storage()._cdata)
        if identity in targets:
            raise ValueError("tensor update targets must not share storage")
        targets.add(identity)
    replacements: dict[int, torch.Tensor] = {}
    for current, replacement in pairs:
        if current is replacement:
            raise ValueError(f"{kind} replacement must use staged storage")
        if current.dtype != replacement.dtype:
            raise TypeError(
                f"{kind} replacement changes dtype from {current.dtype} "
                f"to {replacement.dtype}"
            )
        if current.device != replacement.device:
            raise ValueError(
                f"{kind} replacement changes device from {current.device} "
                f"to {replacement.device}"
            )
        if (
            kind == "content" or id(current) in content_ids
        ) and current.shape != replacement.shape:
            raise ValueError(
                f"content replacement changes shape from {tuple(current.shape)} "
                f"to {tuple(replacement.shape)}"
            )
        if replacement.layout is not torch.strided or not replacement.is_contiguous():
            raise ValueError(f"{kind} replacement must be a contiguous strided tensor")
        if (replacement.device, replacement.untyped_storage()._cdata) in targets:
            raise ValueError(
                f"{kind} replacement must not share storage with any update target"
            )
        replacements[id(current)] = replacement
    return replacements


def _reference_must_resolve_locally(runtime: ModelRuntime, field: Any) -> bool:
    """Return whether a local lookup miss can only mean a dangling reference."""
    if runtime.plan.spatial_world_size == 1 or field.tensor.is_coordinate:
        return True
    schema = runtime.plan.fields.partition.fields
    target = schema.get(field.tensor.references.rsplit(".", 1)[-1])
    if target is not None and target.replicated:
        return True
    return any(
        metadata.partition_by and metadata.partition_by.rsplit(".", 1)[-1] == field.name
        for metadata in schema.values()
    )


def _validate_update_integrity(
    runtime: ModelRuntime,
    fields: Mapping[int, list[tuple[Any, str, Any]]],
    replacements: Mapping[int, torch.Tensor],
) -> None:
    """Re-check staged keys and the references that read or target them.

    Commits are rank-local and may run on a subset of ranks, so no collective
    is used: a miss is rejected whenever it cannot be an off-rank target.
    """

    for identity, matches in fields.items():
        for _module, field_name, metadata in matches:
            if metadata is not None:
                ModuleTensors._validate_key(
                    field_name, metadata, replacements[identity]
                )

    namespace = runtime.namespace
    for module_name in runtime.plan.modules:
        module = runtime.modules[module_name]
        for field in module.spec().tensor_fields.values():
            if (
                field.computed
                or not field.tensor.references
                or not module._is_tensor_field_active(field.name)
            ):
                continue
            values = getattr(module, field.name)
            reference = field.tensor.references
            entry = namespace.get(reference) or namespace.get(
                f"{module_name}.{reference}"
            )
            if entry is None:
                raise ValueError(
                    f"reference {module_name}.{field.name} target {reference!r} "
                    "does not resolve to one opened tensor"
                )
            target = getattr(entry.module, entry.field_name)
            if id(values) not in replacements and id(target) not in replacements:
                continue
            if not isinstance(values, torch.Tensor) or not isinstance(
                target, torch.Tensor
            ):
                continue
            values = replacements.get(id(values), values).reshape(-1)
            target = replacements.get(id(target), target).reshape(-1)
            missing = find_indices_in_torch(values, target.to(values.device)) < 0
            if bool(missing.any()) and _reference_must_resolve_locally(runtime, field):
                raise ValueError(
                    f"update leaves {int(missing.sum().item())} value(s) of "
                    f"reference {module_name}.{field.name} absent from "
                    f"{reference!r}; examples: {values[missing][:5].tolist()}"
                )


def _stale_reference_indices(
    runtime: ModelRuntime,
    replacements: Mapping[int, torch.Tensor],
) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
    """Recompute derived indices whose source or target storage is staged."""
    stale: list[tuple[torch.Tensor, torch.Tensor]] = []
    for module_name in runtime.plan.modules:
        stale.extend(
            runtime.modules[module_name]._stale_reference_indices(replacements)
        )
    return tuple(stale)


def commit_content_update(
    runtime: ModelRuntime,
    bindings: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> StructuralUpdateResult:
    """Copy same-shape tensors; rebuild captures if cached CSR inputs change."""
    return _commit_content_update(runtime, _tensor_pairs(bindings))


def _validate_content_coordinates(fields, replacements, pairs, content_ids) -> None:
    for identity in content_ids:
        matches = fields[identity]
        for _module, field_name, metadata in matches:
            if (
                metadata is not None
                and metadata.is_coordinate
                and not torch.equal(
                    next(current for current, _ in pairs if id(current) == identity),
                    replacements[identity],
                )
            ):
                raise ValueError(
                    f"address-stable content update cannot change coordinate "
                    f"{field_name!r}"
                )


def _commit_content_update(
    runtime: ModelRuntime, pairs: Sequence[tuple[torch.Tensor, torch.Tensor]]
) -> StructuralUpdateResult:
    _require_update_boundary(runtime, "content")
    replacements = _validate_replacements(pairs, kind="content")

    fields = _replacement_fields(runtime, replacements)
    _validate_content_coordinates(fields, replacements, pairs, frozenset(replacements))
    _validate_update_integrity(runtime, fields, replacements)
    derived = _stale_reference_indices(runtime, replacements)
    for current, fresh in derived:
        if (
            current.shape != fresh.shape
            or current.dtype != fresh.dtype
            or current.device != fresh.device
        ):
            raise ValueError(
                "address-stable content update would change the layout of a "
                "derived reference index"
            )

    changed = tuple(current for current, _ in (*pairs, *derived))
    statistics = runtime.statistics
    if runtime.execution.kernel_binding.content_requires_rebind(changed) or (
        statistics is not None and statistics.content_requires_rebind(changed)
    ):
        # Address stability does not imply validity of derived CSR contents.
        # Reuse the full transaction, retaining the caller's content-copy policy.
        return _commit_structural_update(
            runtime, pairs, content_ids=frozenset(replacements)
        )

    bindings = runtime.parameters.prepare_rebind(replacements)
    statistics = runtime.statistics
    selections = (
        () if statistics is None else statistics.prepare_selection_update(replacements)
    )

    mutated = False
    try:
        _synchronize(runtime.plan.device)
        mutated = True
        with torch.inference_mode():
            for current, replacement in (*pairs, *derived, *selections):
                current.copy_(replacement)
        runtime.parameters.install_rebind(bindings)
        return StructuralUpdateResult(
            revision=runtime.execution.structural_revision,
            dimensions=MappingProxyType({}),
            invalidated=False,
        )
    except BaseException as error:
        if mutated:
            runtime.execution.poison(error, phase="address-stable content update")
        raise


def commit_structural_update(
    runtime: ModelRuntime,
    bindings: Iterable[tuple[torch.Tensor, torch.Tensor]],
) -> StructuralUpdateResult:
    """Rebind declared tensors and rebuild every structural consumer.

    Changed symbolic dimensions are inferred from tensor field schemas, so a
    model implementation only stages replacement storage. Every field that
    depends on an inferred dimension must be supplied at its new shape;
    otherwise validation fails before cached execution is invalidated.
    """

    return _commit_structural_update(runtime, _tensor_pairs(bindings))


def _commit_structural_update(
    runtime: ModelRuntime,
    pairs: Sequence[tuple[torch.Tensor, torch.Tensor]],
    *,
    content_ids: frozenset[int] = frozenset(),
) -> StructuralUpdateResult:
    _require_update_boundary(runtime, "structural")
    replacements = _validate_replacements(
        pairs, kind="structural", content_ids=content_ids
    )

    fields = _replacement_fields(runtime, replacements)
    _validate_content_coordinates(fields, replacements, pairs, content_ids)
    inferred = _infer_dimensions(fields, replacements)
    _validate_dependent_shapes(runtime, replacements, inferred)
    _validate_update_integrity(runtime, fields, replacements)
    derived = _stale_reference_indices(runtime, replacements)
    previous_dimensions = {
        key: int(getattr(dimension.owner, dimension.attribute))
        for key, (dimension, _extent) in inferred.items()
    }
    bindings = runtime.parameters.prepare_rebind(replacements)
    statistics = runtime.statistics
    selections = (
        () if statistics is None else statistics.prepare_selection_update(replacements)
    )

    mutated = False
    try:
        _synchronize(runtime.plan.device)
        runtime.execution.invalidate()
        mutated = True
        with torch.inference_mode():
            for current, replacement in pairs:
                if id(current) in content_ids:
                    current.copy_(replacement)
                else:
                    current.set_(replacement)
            for current, fresh in (*derived, *selections):
                if current.shape == fresh.shape:
                    current.copy_(fresh)
                else:
                    current.set_(fresh)
        changes = _publish_dimensions(inferred, previous_dimensions)
        _verify_dimensions(inferred)

        if statistics is not None:
            statistics.recompile_resized_sources()

        runtime.checkpoint = CheckpointRuntime(runtime)
        runtime.parameters.install_rebind(bindings)
        runtime.execution.structural_revision += 1
        return StructuralUpdateResult(
            revision=runtime.execution.structural_revision,
            dimensions=MappingProxyType(changes),
        )
    except BaseException as error:
        if mutated:
            runtime.execution.poison(error, phase="structural tensor update")
        raise

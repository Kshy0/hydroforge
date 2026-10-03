"""Coordinate ownership graph of the active tensor fields."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING

from hydroforge.contracts.fields import PartitionSchema, TensorMetadata

if TYPE_CHECKING:
    from hydroforge.compiler.fields import FieldEntry


def bare(name: str | None) -> str | None:
    """Drop a ``module.`` qualifier from a declared field reference."""

    return name.rsplit(".", 1)[-1] if name else None


def coordinate_identity(name: str, entries: Iterable[FieldEntry]) -> str:
    """Resolve a coordinate to its unique owner before comparing schemas."""
    candidates = {
        entry.qualified: entry
        for entry in entries
        if entry.tensor.is_coordinate
        and (entry.qualified == name if "." in name else entry.name == name)
    }
    if len(candidates) != 1:
        raise ValueError(f"coordinate {name!r} has no unique declared owner")
    return next(iter(candidates))


def _leading_dimension(shape: tuple[str | int, ...]) -> str | int | None:
    """Compare symbolic axes; actual owner extents are checked at binding."""
    if not shape:
        return None
    token = shape[0]
    return bare(token) if isinstance(token, str) else token


def coordinate_is_partitioned(
    schema: PartitionSchema, partition_key: str | None, coordinate: str
) -> bool:
    """Return whether a coordinate's lineage reaches the partition key."""

    fields = schema.fields
    while coordinate != partition_key:
        metadata = fields[coordinate]
        if metadata.replicated:
            return False
        via = bare(metadata.partition_by)
        target = bare(fields[via].references) if via else bare(metadata.references)
        if target is None:
            return False
        coordinate = target
    return True


def _schema(
    entries: Iterable[FieldEntry], partition_key: str | None, modules=None
) -> PartitionSchema:
    entries = tuple(entries)
    fields: dict[str, TensorMetadata] = {}
    computed: list[FieldEntry] = []
    coordinate_owners: dict[str, list[str]] = {}
    for entry in entries:
        if entry.computed:
            computed.append(entry)
            continue
        fields[entry.name] = entry.tensor
        if entry.tensor.is_coordinate:
            coordinate_owners.setdefault(entry.name, []).append(entry.module)
    for name, owners in coordinate_owners.items():
        if len(owners) > 1:
            raise ValueError(
                f"Coordinate '{name}' is declared by modules {owners}; a "
                "coordinate names one model axis, so declare it in one "
                "module and reference it from the others."
            )
    coordinates = {name for name, metadata in fields.items() if metadata.is_coordinate}
    if modules is not None:
        for entry in entries:
            for token in (
                entry.tensor.dim_coords,
                entry.tensor.references,
                entry.tensor.selects,
            ):
                if token:
                    coordinate_identity(token, entries)
    # Output coordinates and selections resolve by unqualified name, which
    # an expression virtual would otherwise own.
    shadowing = sorted(
        entry.qualified
        for entry in computed
        if entry.name in coordinates and entry.expression_virtual
    )
    if shadowing:
        raise ValueError(
            f"Expression virtual fields {shadowing} reuse a coordinate name; "
            "rename them so output coordinates resolve to their owner."
        )
    selections: dict[str, str] = {}

    # Structured-grid models without logical CoordinateField axes are
    # unpartitioned by construction; they need no model-side override.
    if not coordinates:
        return PartitionSchema(
            fields=MappingProxyType(fields),
            coordinates=frozenset(),
            selections=MappingProxyType({}),
        )

    if partition_key is None:
        raise ValueError("Model partition_key must be configured.")
    if partition_key not in coordinates:
        raise ValueError(f"partition_key '{partition_key}' must be a CoordinateField.")
    for name, metadata in (
        *fields.items(),
        *((entry.name, entry.tensor) for entry in computed),
    ):
        coordinate = bare(metadata.dim_coords)
        if not coordinate:
            continue
        if coordinate not in coordinates:
            raise ValueError(
                f"Field '{name}' uses dim_coords='{coordinate}', but it is "
                "not a CoordinateField."
            )
        axis = _leading_dimension(metadata.shape)
        expected = _leading_dimension(fields[coordinate].shape)
        if axis != expected:
            raise ValueError(
                f"Field '{name}' uses dim_coords='{coordinate}', but its "
                f"first dimension {axis!r} is not the coordinate dimension "
                f"{expected!r}."
            )
    for name, metadata in fields.items():
        references = bare(metadata.references)
        if references and references not in coordinates:
            raise ValueError(
                f"Field '{name}' references unknown coordinate '{references}'."
            )
        selects = bare(metadata.selects)
        if selects:
            if name not in coordinates:
                raise ValueError(f"Selection '{name}' must be a CoordinateField.")
            if references != selects:
                raise ValueError(
                    f"Selection '{name}' must reference the coordinate it "
                    f"selects ('{selects}')."
                )
            previous = selections.get(selects)
            if previous is not None:
                raise ValueError(
                    f"Coordinate '{selects}' has multiple default selections: "
                    f"'{previous}' and '{name}'."
                )
            selections[selects] = name

        partition_by = bare(metadata.partition_by)
        if metadata.replicated and name not in coordinates:
            raise ValueError(
                f"replicated=True is only valid on CoordinateField, got '{name}'."
            )
        if metadata.replicated and (
            name == partition_key or partition_by or references
        ):
            raise ValueError(
                f"Replicated coordinate '{name}' cannot define partition lineage."
            )
        if (
            name in coordinates
            and name != partition_key
            and not partition_by
            and not references
            and not metadata.replicated
        ):
            raise ValueError(
                f"Coordinate '{name}' has no ownership lineage. Declare "
                "partition_by/references or set replicated=True."
            )
        if partition_by:
            if name not in coordinates:
                raise ValueError(
                    f"partition_by is only valid on CoordinateField, got '{name}'."
                )
            via = fields.get(partition_by)
            if via is None:
                raise ValueError(
                    f"Coordinate '{name}' uses partition_by='{partition_by}', "
                    "but it is not an active tensor field of the opened modules."
                )
            if bare(via.dim_coords) != name:
                raise ValueError(
                    f"Partition field '{partition_by}' must be aligned to "
                    f"coordinate '{name}', got dim_coords={via.dim_coords!r}."
                )
            if not via.references:
                raise ValueError(
                    f"Partition field '{partition_by}' must declare references."
                )

    lineage: dict[str, str] = {}
    for coordinate in coordinates:
        if coordinate == partition_key:
            continue
        metadata = fields[coordinate]
        via = bare(metadata.partition_by)
        target = bare(fields[via].references) if via else bare(metadata.references)
        if target is not None:
            lineage[coordinate] = target
    for origin in lineage:
        seen: set[str] = set()
        coordinate = origin
        while coordinate in lineage:
            if coordinate in seen:
                raise ValueError(
                    "partition coordinate lineage must be acyclic; "
                    f"cycle includes {coordinate!r}"
                )
            seen.add(coordinate)
            coordinate = lineage[coordinate]

    return PartitionSchema(
        fields=MappingProxyType(fields),
        coordinates=frozenset(coordinates),
        selections=MappingProxyType(selections),
    )


def compile_partition(
    entries: Iterable[FieldEntry], partition_key: str | None, *, modules=None
) -> tuple[PartitionSchema, Mapping[str, str]]:
    """Validate the partition graph and group each field by its coordinate."""

    schema = _schema(entries, partition_key, modules)
    groups: dict[str, str] = {}
    for name, metadata in schema.fields.items():
        if name in schema.coordinates:
            if coordinate_is_partitioned(schema, partition_key, name):
                groups[name] = name
            continue
        coordinate = bare(metadata.dim_coords)
        if coordinate and coordinate_is_partitioned(schema, partition_key, coordinate):
            groups[name] = coordinate
    return schema, MappingProxyType(groups)

"""Owned process groups for independent ensemble and spatial partition axes."""

from __future__ import annotations

from typing import Any, Literal

import torch.distributed as dist
from pydantic import Field, PrivateAttr, model_validator

from hydroforge.contracts.errors import ResourceCleanupError, cleanup_on_exit
from hydroforge.contracts.validation import HydroForgeModel

ParallelAxis = Literal["spatial", "ensemble"]


class EnsembleParallel(HydroForgeModel):
    """A row-major ensemble × spatial process mesh, owned by a context manager.

    Initialize distributed execution first. Every world rank must enter the same
    declaration, keep it alive until its models close, and exit it exactly once.
    The default process group belongs to the caller and is never destroyed here.
    """

    ensemble_size: int = Field(ge=1)
    ensemble_partitions: int = Field(default=1, ge=1)
    spatial_partitions: int = Field(default=1, ge=1)

    _rank: int = PrivateAttr(default=0)
    _default_group: Any = PrivateAttr(default=None)
    _spatial_group: Any = PrivateAttr(default=None)
    _ensemble_group: Any = PrivateAttr(default=None)
    _owned_groups: tuple[Any, ...] = PrivateAttr(default=())
    _entered: bool = PrivateAttr(default=False)
    _closed: bool = PrivateAttr(default=False)

    @model_validator(mode="after")
    def _validate_members(self):
        if self.ensemble_partitions > self.ensemble_size:
            raise ValueError("ensemble partitions cannot exceed the number of members")
        return self

    @property
    def world_size(self) -> int:
        return self.ensemble_partitions * self.spatial_partitions

    @property
    def ensemble_rank(self) -> int:
        return self._rank // self.spatial_partitions

    @property
    def spatial_rank(self) -> int:
        return self._rank % self.spatial_partitions

    @property
    def member_slice(self) -> slice:
        width, remainder = divmod(self.ensemble_size, self.ensemble_partitions)
        start = self.ensemble_rank * width + min(self.ensemble_rank, remainder)
        return slice(start, start + width + int(self.ensemble_rank < remainder))

    @property
    def local_ensemble_size(self) -> int:
        selection = self.member_slice
        return selection.stop - selection.start

    @property
    def member_ids(self) -> tuple[int, ...]:
        selection = self.member_slice
        return tuple(range(selection.start, selection.stop))

    def rank_groups(self, axis: ParallelAxis) -> tuple[tuple[int, ...], ...]:
        if axis == "spatial":
            return tuple(
                tuple(range(start, start + self.spatial_partitions))
                for start in range(0, self.world_size, self.spatial_partitions)
            )
        return tuple(
            tuple(range(offset, self.world_size, self.spatial_partitions))
            for offset in range(self.spatial_partitions)
        )

    def group(self, axis: ParallelAxis):
        self.validate_live()
        return self._spatial_group if axis == "spatial" else self._ensemble_group

    def validate_live(self) -> None:
        if not self._entered or self._closed:
            raise RuntimeError(
                "ensemble process mesh must be inside its active context"
            )
        initialized = dist.is_available() and dist.is_initialized()
        observed = dist.group.WORLD if initialized else None
        if observed is not self._default_group:
            raise RuntimeError("ensemble process mesh lost its default process group")
        if initialized and (
            dist.get_rank() != self._rank or dist.get_world_size() != self.world_size
        ):
            raise RuntimeError("ensemble process mesh topology changed")

    def __enter__(self):
        if self._entered or self._closed:
            raise RuntimeError("ensemble process meshes cannot be re-entered")
        initialized = dist.is_available() and dist.is_initialized()
        world_size = dist.get_world_size() if initialized else 1
        declaration = (
            self.ensemble_size,
            self.ensemble_partitions,
            self.spatial_partitions,
        )
        if initialized:
            observed = [None] * world_size
            dist.all_gather_object(observed, declaration)
            if any(value != declaration for value in observed):
                raise ValueError(
                    "ensemble process mesh declarations differ across ranks"
                )
        if world_size != self.world_size:
            raise ValueError(
                "ensemble × spatial partitions must equal the process world size"
            )
        self._rank = dist.get_rank() if initialized else 0
        self._default_group = dist.group.WORLD if initialized else None
        self._entered = True
        try:
            for axis in ("spatial", "ensemble"):
                groups = self.rank_groups(axis)
                if len(groups) == 1 or len(groups[0]) == 1:
                    continue
                for ranks in groups:
                    group = dist.new_group(ranks=list(ranks))
                    if self._rank in ranks:
                        self._owned_groups = (*self._owned_groups, group)
                        if axis == "spatial":
                            self._spatial_group = group
                        else:
                            self._ensemble_group = group
        except BaseException as primary:
            try:
                self.close()
            except BaseException as cleanup:
                raise ResourceCleanupError(
                    "ensemble process mesh construction",
                    (primary, cleanup),
                ) from primary
            raise
        return self

    def close(self) -> None:
        groups, self._owned_groups = self._owned_groups, ()
        self._closed = True
        self._spatial_group = None
        self._ensemble_group = None
        with cleanup_on_exit(
            "ensemble process groups",
            tuple(
                lambda group=group: dist.destroy_process_group(group)
                for group in reversed(groups)
            ),
        ):
            pass

    def __exit__(self, error_type, error, traceback) -> bool:
        try:
            self.close()
        except BaseException as cleanup:
            if error is not None:
                raise ResourceCleanupError(
                    "ensemble process mesh context",
                    (error, cleanup),
                ) from error
            raise
        return False

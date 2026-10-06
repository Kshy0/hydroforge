# LICENSE HEADER MANAGED BY add-license-header
# Copyright (c) 2025 Shengyu Kang (Wuhan University)
# Licensed under the Apache License, Version 2.0
# http://www.apache.org/licenses/LICENSE-2.0
#

"""The public ``AbstractModel``: declaration, compilation and runtime ownership."""

from __future__ import annotations

from abc import ABC
from collections.abc import Mapping
from datetime import datetime
from types import MappingProxyType, TracebackType
from typing import TYPE_CHECKING, Any, Self

import cftime

from hydroforge.compiler.plan import ModelPlan, compile_model
from hydroforge.core.errors import ResourceCleanupError
from hydroforge.data.input import InputProxy
from hydroforge.declare.model import ModelDeclaration
from hydroforge.execution.context import managed_step_active
from hydroforge.execution.session import ModelRuntime
from hydroforge.model.results import ModelResults

if TYPE_CHECKING:
    from hydroforge.execution.partition import GroupRankLookup


class AbstractModel(ModelDeclaration, ABC):
    """Generic master controller for hydroforge models.

    Construction validates and compiles the declaration without I/O.
    ``materialize()`` (or entering ``with model:``) loads inputs, constructs
    modules and starts output; managed steps, between-step APIs,
    ``save_state()`` and ``update_structure()`` materialize a declared model
    on first use. ``close()`` releases every runtime resource.
    """

    def model_post_init(self, context: Any) -> None:
        """Compile the declaration and create the runtime that will own it."""

        del context
        plan = compile_model(type(self).spec(), self)
        self._plan = plan
        self._runtime = ModelRuntime(self, plan)

    @property
    def plan(self) -> ModelPlan:
        """Every compile-time decision of this model."""

        return self.__pydantic_private__["_plan"]

    @property
    def results(self) -> ModelResults:
        """Statistics results declared by ``OutputConfig.variables``."""

        return ModelResults(self)

    @property
    def current_time(self) -> datetime | cftime.datetime | None:
        """Physical date of the next step, or the end of a completed schedule."""

        runtime = ModelRuntime.of(self)
        if runtime.state == "closed":
            raise runtime.unavailable(f"{type(self).__name__}.current_time")
        return runtime.clock.current_time

    @property
    def schedule_index(self) -> int | None:
        """Next execution index, or None for a model without a schedule."""

        runtime = ModelRuntime.of(self)
        if runtime.state == "closed":
            raise runtime.unavailable(f"{type(self).__name__}.schedule_index")
        return None if self.plan.schedule is None else runtime.clock.schedule_index

    @property
    def group_id_to_rank(self) -> GroupRankLookup:
        runtime = ModelRuntime.of(self)
        runtime.require_materialized(f"{type(self).__name__}.group_id_to_rank")
        return runtime.partition.group_ranks

    @property
    def parameter_dependencies(self) -> Mapping[str, tuple[str, ...]]:
        """Observed derived-field inputs; empty until the first parameter event.

        Inspection does not initialize the runtime or discover dependencies.
        The immutable graph applies to the structure at its last discovery.
        """
        parameters = ModelRuntime.of(self).parameters
        return MappingProxyType({}) if parameters is None else parameters.dependencies

    def materialize(self) -> Self:
        """Load inputs and construct modules; collective across ranks.

        Idempotent while materialized; after ``close()`` it builds a new
        runtime from the declared initial state.
        """

        ModelRuntime.of(self).materialize()
        return self

    def close(self) -> None:
        """Release every runtime resource; only ``materialize()`` rebuilds."""

        ModelRuntime.of(self).close()

    def __enter__(self) -> Self:
        return self.materialize()

    def __exit__(
        self,
        error_type: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        if error is None:
            self.close()
            return
        try:
            self.close()
        except BaseException as close_error:
            raise ResourceCleanupError(
                "model after a failure", (error, close_error)
            ) from error

    def prepare_model_input(self, values: dict[str, Any]) -> dict[str, Any]:
        """Return this rank's module input values before modules are built.

        ``values`` holds every present input field, already rank-sliced.
        Override to add model-derived construction values; the default
        returns them unchanged.
        """

        return values

    def initialize_model_state(self) -> None:
        """Initialize ordered model state inside HydroForge's transaction.

        Model subclasses own the complete initialization order.  This hook is
        called after all modules have been constructed and tensor modes applied,
        so a controller can explicitly sequence cross-module cold starts and
        workspace materialization inside HydroForge's transaction.
        """

    def release_model_state(self) -> None:
        """Release external resources created by ``initialize_model_state``.

        Called once the model's execution resources are closed, both on
        ``close()`` and when a materialization fails after
        ``initialize_model_state`` was entered.
        """

    def update_structure(self):
        """Run the ordered module structure pass between managed steps."""

        return ModelRuntime.of(self).update_structure()

    def save_state(self) -> InputProxy:
        """Publish a complete construction input at the committed clock.

        Returns the published checkpoint as a lazy ``InputProxy`` on every
        rank; its ``file_path`` names the file.
        """

        if managed_step_active():
            raise RuntimeError("save_state is allowed only between managed steps")
        error: ValueError | None = None
        if self.ensemble_size is not None:
            error = ValueError(
                "checkpoint save currently requires a non-ensemble model"
            )
        elif self.plan.output.directory is None:
            error = ValueError("save_state requires OutputConfig(dir=...)")
        runtime = ModelRuntime.of(self)
        runtime.channel.preflight(
            error,
            phase="checkpoint.save.api-validation",
            scope="distributed checkpoint save entry validation",
        )
        runtime.ensure_materialized(f"{type(self).__name__}.save_state")
        return runtime.checkpoint.save()

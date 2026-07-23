"""Immutable results of compiling one model specialization."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import torch


@dataclass(frozen=True, slots=True)
class FieldOwner:
    module_name: str
    field_name: str
    owner: Any


@dataclass(frozen=True, slots=True)
class FieldNamespace:
    owners: Mapping[str, tuple[FieldOwner, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(self, "owners", MappingProxyType(dict(self.owners)))


@dataclass(frozen=True, slots=True)
class ModulePlan:
    order: tuple[str, ...]
    dependencies: Mapping[str, tuple[str, ...]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "dependencies", MappingProxyType(dict(self.dependencies)),
        )


@dataclass(frozen=True, slots=True)
class RuntimePlan:
    backend: str
    device: torch.device
    capture_mode: str


@dataclass(frozen=True, slots=True)
class KernelPlan:
    fields: FieldNamespace


@dataclass(frozen=True, slots=True)
class ExecutionPlan:
    policy_count: int = 0


@dataclass(frozen=True, slots=True)
class StatisticsPlan:
    enabled: bool = False
    variables: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ModelPlan:
    modules: ModulePlan
    capabilities: frozenset[str]
    fields: FieldNamespace
    runtime: RuntimePlan
    kernels: KernelPlan
    execution: ExecutionPlan = ExecutionPlan()
    statistics: StatisticsPlan = StatisticsPlan()

    def has_module(self, name: str) -> bool:
        return name in self.modules.order

    def has_feature(self, name: str) -> bool:
        return name in self.capabilities

__all__ = [
    "ExecutionPlan", "FieldNamespace", "FieldOwner", "KernelPlan",
    "ModelPlan", "ModulePlan", "RuntimePlan", "StatisticsPlan",
]

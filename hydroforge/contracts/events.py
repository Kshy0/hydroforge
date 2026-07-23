"""Structured model events with one configurable output boundary."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol, runtime_checkable


@dataclass(frozen=True)
class ModelEvent:
    level: str
    name: str
    message: str
    fields: dict[str, Any]


@runtime_checkable
class EventSink(Protocol):
    def emit(self, event: ModelEvent) -> None: ...


class ConsoleEventSink:
    """Default event sink; applications may replace it during construction."""

    def __init__(self, writer: Callable[[str], None] = print) -> None:
        self.writer = writer

    def emit(self, event: ModelEvent) -> None:
        if event.name == "model.memory":
            rank = event.fields["rank"]
            modules = event.fields["modules"]
            total_mb = event.fields["total_mb"]
            self.writer(f"\n[rank {rank}] Memory Usage Summary:")
            self.writer(f"{'Module':<30} | {'Memory (MB)':<15}")
            self.writer("-" * 48)
            for name, memory_mb in modules.items():
                self.writer(f"{name:<30} | {memory_mb:<15.2f}")
            self.writer("-" * 48)
            self.writer(f"{'Total':<30} | {total_mb:<15.2f} MB\n")
            return
        if event.name == "step.completed":
            current_time = event.fields.get("current_time")
            substeps = event.fields.get("adaptive_time_step")
            message = f"Processed step at {current_time}"
            if substeps is not None:
                message += f", adaptive_time_step={substeps}"
            progress = event.fields.get("progress")
            if progress:
                message += f" | {progress}"
            if self.writer is print:
                print(f"\r\033[K{message}", end="", flush=True)
            else:
                self.writer(message)
            return
        fields = " ".join(f"{key}={value}" for key, value in event.fields.items())
        suffix = f" | {fields}" if fields else ""
        self.writer(f"[{event.level}] {event.message}{suffix}")


class NullEventSink:
    def emit(self, event: ModelEvent) -> None:
        del event


def emit(
    model: Any,
    level: str,
    name: str,
    message: str,
    **fields: Any,
) -> None:
    model.event_sink.emit(ModelEvent(level, name, message, fields))

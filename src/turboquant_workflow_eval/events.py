"""Synchronous event system for study lifecycle notifications."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class StudyEvent:
    """A single study lifecycle event."""

    kind: str  # "prompt_started", "prompt_completed", "policy_started", "policy_completed",
                # "study_started", "study_completed", "error", "early_stop"
    timestamp: float = field(default_factory=time.time)
    data: dict[str, Any] = field(default_factory=dict)


class EventBus:
    """Synchronous publish/subscribe event bus.

    All subscribers are called inline (no threading) — the bottleneck is GPU
    inference, not event dispatch.
    """

    def __init__(self) -> None:
        self._subscribers: list[Callable[[StudyEvent], None]] = []

    def subscribe(self, callback: Callable[[StudyEvent], None]) -> None:
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[StudyEvent], None]) -> None:
        self._subscribers = [s for s in self._subscribers if s is not callback]

    def emit(self, event: StudyEvent) -> None:
        for callback in self._subscribers:
            try:
                callback(event)
            except Exception:
                pass  # subscribers must not break the study loop

    def emit_new(self, kind: str, **data: Any) -> None:
        """Convenience: create and emit a StudyEvent in one call."""
        self.emit(StudyEvent(kind=kind, data=data))

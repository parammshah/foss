"""
Event publisher and learning event models for Reserve Juggler.

Allows Team 3 (Audit, XAI, and Feedback Store) to observe decision proposals,
rejected actions, hard-gate evaluations, and execution outcomes.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Protocol, Sequence
from uuid import uuid4

from reserve_juggler.decision.protocols import LearningEvent


@dataclass(frozen=True)
class StandardLearningEvent:
    """Standardized learning event implementation."""

    versioned_event_type: str
    source_module: str
    correlation_id: str
    structured_data: dict[str, Any]
    model_policy_versions: dict[str, str]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    event_id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """Convert event to JSON-serializable dictionary."""
        return {
            "event_id": self.event_id,
            "versioned_event_type": self.versioned_event_type,
            "source_module": self.source_module,
            "timestamp": self.timestamp.isoformat(),
            "correlation_id": self.correlation_id,
            "structured_data": self.structured_data,
            "model_policy_versions": self.model_policy_versions,
        }

    def to_json(self) -> str:
        """Convert event to JSON string."""
        return json.dumps(self.to_dict())


class EventPublisher(Protocol):
    """Protocol for event publishing."""

    def publish(self, event: LearningEvent) -> None:
        """Publish a learning event."""
        ...


class InMemoryEventPublisher:
    """In-memory event publisher with callback support for Team 3 integration."""

    def __init__(self, callbacks: Sequence[Callable[[LearningEvent], None]] | None = None) -> None:
        self.events: list[LearningEvent] = []
        self._callbacks: list[Callable[[LearningEvent], None]] = list(callbacks or [])

    def register_callback(self, callback: Callable[[LearningEvent], None]) -> None:
        """Register a callback for downstream subscribers."""
        self._callbacks.append(callback)

    def publish(self, event: LearningEvent) -> None:
        """Publish event to internal history and trigger registered callbacks."""
        self.events.append(event)
        for cb in self._callbacks:
            cb(event)

    def clear(self) -> None:
        """Clear recorded events."""
        self.events.clear()

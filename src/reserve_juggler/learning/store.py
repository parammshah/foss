"""
Shared Event Store for System-Wide Self-Learning.
Records versioned inputs, forecasts, decisions, outcomes, evaluation metrics,
and human feedback with correlation IDs and model/policy lineage.
"""

from __future__ import annotations
import json
import logging
import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from reserve_juggler.integration.contracts import LearningEvent

logger = logging.getLogger(__name__)


class EventStore:
    """
    Append-only versioned event store for Reserve Juggler.
    Allows querying by correlation_id, subsystem module, and timestamp range.
    """

    def __init__(self, persistence_file: Optional[str] = None):
        self.persistence_file = persistence_file
        self._events: List[LearningEvent] = []
        self._by_correlation: Dict[str, List[LearningEvent]] = {}
        self._by_module: Dict[str, List[LearningEvent]] = {}
        
        if self.persistence_file and os.path.exists(self.persistence_file):
            self._load_from_disk()

    def record(self, event: LearningEvent) -> None:
        """Record a versioned LearningEvent."""
        self._events.append(event)
        
        # Index by correlation ID
        if event.correlation_id:
            self._by_correlation.setdefault(event.correlation_id, []).append(event)
            
        # Index by module
        self._by_module.setdefault(event.module, []).append(event)

        # Append to persistent storage if configured
        if self.persistence_file:
            self._append_to_disk(event)

    def get_by_correlation_id(self, correlation_id: str) -> List[LearningEvent]:
        """Retrieve all events related to a single decision lifecycle."""
        return self._by_correlation.get(correlation_id, [])

    def get_by_module(self, module: str) -> List[LearningEvent]:
        """Retrieve all events from a given subsystem module."""
        return self._by_module.get(module, [])

    def get_all(self) -> List[LearningEvent]:
        """Return all recorded events in chronological order."""
        return list(self._events)

    def get_in_time_range(
        self, start_iso: Optional[str] = None, end_iso: Optional[str] = None
    ) -> List[LearningEvent]:
        """Query events between start_iso and end_iso."""
        events = self._events
        if start_iso:
            events = [e for e in events if e.timestamp >= start_iso]
        if end_iso:
            events = [e for e in events if e.timestamp <= end_iso]
        return events

    def clear(self) -> None:
        """Clear the store (used for test isolation)."""
        self._events.clear()
        self._by_correlation.clear()
        self._by_module.clear()
        if self.persistence_file and os.path.exists(self.persistence_file):
            try:
                os.remove(self.persistence_file)
            except Exception as e:
                logger.warning("Failed to remove persistence file: %s", str(e))

    def _append_to_disk(self, event: LearningEvent) -> None:
        try:
            line = json.dumps({
                "event_id": event.event_id,
                "versioned_event_type": event.versioned_event_type,
                "module": event.module,
                "timestamp": event.timestamp,
                "correlation_id": event.correlation_id,
                "payload": event.payload,
                "associated_versions": event.associated_versions,
            })
            with open(self.persistence_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception as e:
            logger.error("Failed to append event to disk: %s", str(e))

    def _load_from_disk(self) -> None:
        try:
            with open(self.persistence_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    ev = LearningEvent(
                        event_id=data["event_id"],
                        versioned_event_type=data["versioned_event_type"],
                        module=data["module"],
                        timestamp=data["timestamp"],
                        correlation_id=data["correlation_id"],
                        payload=data["payload"],
                        associated_versions=data.get("associated_versions", {}),
                    )
                    self._events.append(ev)
                    if ev.correlation_id:
                        self._by_correlation.setdefault(ev.correlation_id, []).append(ev)
                    self._by_module.setdefault(ev.module, []).append(ev)
        except Exception as e:
            logger.error("Failed to load events from disk: %s", str(e))

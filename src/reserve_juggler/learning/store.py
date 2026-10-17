"""
Shared Event Store for System-Wide Self-Learning.
Wraps SqliteEventStore for transactional, durable persistence with zero silent write swallowing.
"""

from __future__ import annotations
import logging
from typing import Any, Dict, List, Optional
from reserve_juggler.contracts import LearningEvent
from reserve_juggler.learning.sqlite_store import PersistenceError, SqliteEventStore

logger = logging.getLogger(__name__)


class EventStore:
    """
    Append-only versioned event store for Reserve Juggler.
    Allows querying by correlation_id, subsystem module, and timestamp range.
    Uses SQLite backend for durable transactions.
    """

    def __init__(self, persistence_file: Optional[str] = None):
        self.persistence_file = persistence_file or ":memory:"
        self.sqlite_store = SqliteEventStore(db_path=self.persistence_file)

    def record(self, event: LearningEvent) -> None:
        """Record a versioned LearningEvent. Raises PersistenceError if write fails."""
        self.sqlite_store.record_event(event)

    def get_by_correlation_id(self, correlation_id: str) -> List[LearningEvent]:
        """Retrieve all events related to a single decision or forecast lifecycle."""
        return self.sqlite_store.get_by_correlation_id(correlation_id)

    def get_by_module(self, module: str) -> List[LearningEvent]:
        """Retrieve all events from a given subsystem module."""
        return self.sqlite_store.get_by_module(module)

    def get_all(self) -> List[LearningEvent]:
        """Return all recorded events in chronological order."""
        return self.sqlite_store.get_all_events()

    def get_in_time_range(
        self, start_iso: Optional[str] = None, end_iso: Optional[str] = None
    ) -> List[LearningEvent]:
        """Query events between start_iso and end_iso."""
        events = self.sqlite_store.get_all_events()
        if start_iso:
            events = [e for e in events if e.timestamp >= start_iso]
        if end_iso:
            events = [e for e in events if e.timestamp <= end_iso]
        return events

    def clear(self) -> None:
        """Clear all events (used for test isolation)."""
        self.sqlite_store.clear()

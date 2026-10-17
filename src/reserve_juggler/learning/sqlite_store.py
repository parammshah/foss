"""
Durable SQLite Event and Governance Store.
Provides transactional, append-oriented persistence for LearningEvents,
CandidateUpdates, and VersionAudit entries with zero silent failure swallowing.
"""

from __future__ import annotations
import json
import logging
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from reserve_juggler.contracts import LearningEvent
from reserve_juggler.learning.models import CandidateStatus, CandidateUpdate, CandidateValidationReport

logger = logging.getLogger(__name__)


class PersistenceError(Exception):
    """Raised when an operation fails to persist to durable storage."""
    pass


class SqliteEventStore:
    """
    Durable, queryable SQLite event and audit store.
    Supports in-memory mode (':memory:') for isolated unit tests
    and file-backed mode (e.g. 'reserve_juggler.db') for durable operation.
    """

    def __init__(self, db_path: str = ":memory:"):
        self.db_path = db_path
        self._shared_conn: Optional[sqlite3.Connection] = None
        if self.db_path == ":memory:":
            self._shared_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_conn.row_factory = sqlite3.Row
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        try:
            conn = sqlite3.connect(self.db_path, timeout=10.0, check_same_thread=False)
            conn.row_factory = sqlite3.Row
            return conn
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to connect to database at {self.db_path}: {e}") from e

    def _init_db(self) -> None:
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS learning_events (
                        event_id TEXT PRIMARY KEY,
                        versioned_event_type TEXT NOT NULL,
                        module TEXT NOT NULL,
                        timestamp TEXT NOT NULL,
                        correlation_id TEXT,
                        payload_json TEXT NOT NULL,
                        associated_versions_json TEXT NOT NULL
                    );
                    """
                )
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_corr ON learning_events(correlation_id);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_module ON learning_events(module);")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_events_ts ON learning_events(timestamp);")

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS candidate_updates (
                        candidate_id TEXT PRIMARY KEY,
                        component TEXT NOT NULL,
                        parameter_key TEXT NOT NULL,
                        current_value_json TEXT,
                        proposed_value_json TEXT,
                        description TEXT,
                        author TEXT,
                        created_at TEXT,
                        status TEXT,
                        validation_report_json TEXT,
                        holdout_score REAL,
                        holdout_sample_size INTEGER,
                        stress_shortfalls INTEGER,
                        safety_passed INTEGER,
                        promoted_at TEXT,
                        promoted_by TEXT,
                        promotion_role TEXT,
                        rejection_reason TEXT,
                        is_metadata_only INTEGER
                    );
                    """
                )

                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS version_audits (
                        entry_id TEXT PRIMARY KEY,
                        component TEXT NOT NULL,
                        previous_version TEXT,
                        new_version TEXT,
                        timestamp TEXT NOT NULL,
                        author TEXT,
                        approver TEXT,
                        approver_role TEXT,
                        action_type TEXT,
                        justification TEXT
                    );
                    """
                )
                conn.commit()
        except sqlite3.Error as e:
            raise PersistenceError(f"Database initialization failed: {e}") from e

    def record_event(self, event: LearningEvent) -> None:
        """Records a versioned LearningEvent. Raises PersistenceError on failure."""
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO learning_events (
                        event_id, versioned_event_type, module, timestamp, correlation_id,
                        payload_json, associated_versions_json
                    ) VALUES (?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        event.event_id,
                        event.versioned_event_type,
                        event.module,
                        event.timestamp,
                        event.correlation_id,
                        json.dumps(event.payload),
                        json.dumps(event.associated_versions),
                    ),
                )
                conn.commit()
        except sqlite3.Error as e:
            logger.error("Durable event persistence failed for event %s: %s", event.event_id, str(e))
            raise PersistenceError(f"Failed to record event {event.event_id}: {e}") from e

    def get_all_events(self) -> List[LearningEvent]:
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT * FROM learning_events ORDER BY timestamp ASC;")
                events = []
                for row in cursor.fetchall():
                    events.append(
                        LearningEvent(
                            event_id=row["event_id"],
                            versioned_event_type=row["versioned_event_type"],
                            module=row["module"],
                            timestamp=row["timestamp"],
                            correlation_id=row["correlation_id"],
                            payload=json.loads(row["payload_json"]),
                            associated_versions=json.loads(row["associated_versions_json"]),
                        )
                    )
                return events
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to query events: {e}") from e

    def get_by_correlation_id(self, correlation_id: str) -> List[LearningEvent]:
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM learning_events WHERE correlation_id = ? ORDER BY timestamp ASC;",
                    (correlation_id,),
                )
                return [
                    LearningEvent(
                        event_id=row["event_id"],
                        versioned_event_type=row["versioned_event_type"],
                        module=row["module"],
                        timestamp=row["timestamp"],
                        correlation_id=row["correlation_id"],
                        payload=json.loads(row["payload_json"]),
                        associated_versions=json.loads(row["associated_versions_json"]),
                    )
                    for row in cursor.fetchall()
                ]
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to query correlation {correlation_id}: {e}") from e

    def get_by_module(self, module: str) -> List[LearningEvent]:
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM learning_events WHERE module = ? ORDER BY timestamp ASC;",
                    (module,),
                )
                return [
                    LearningEvent(
                        event_id=row["event_id"],
                        versioned_event_type=row["versioned_event_type"],
                        module=row["module"],
                        timestamp=row["timestamp"],
                        correlation_id=row["correlation_id"],
                        payload=json.loads(row["payload_json"]),
                        associated_versions=json.loads(row["associated_versions_json"]),
                    )
                    for row in cursor.fetchall()
                ]
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to query module {module}: {e}") from e

    def save_candidate(self, candidate: CandidateUpdate) -> None:
        try:
            with self._get_connection() as conn:
                v_report = None
                if candidate.validation_report:
                    v_report = json.dumps({
                        "candidate_id": candidate.validation_report.candidate_id,
                        "component": candidate.validation_report.component,
                        "validation_dataset": candidate.validation_report.validation_dataset,
                        "validation_sample_size": candidate.validation_report.validation_sample_size,
                        "champion_metric": candidate.validation_report.champion_metric,
                        "candidate_metric": candidate.validation_report.candidate_metric,
                        "metric_name": candidate.validation_report.metric_name,
                        "metric_improved": candidate.validation_report.metric_improved,
                        "stress_shortfall_count": candidate.validation_report.stress_shortfall_count,
                        "safety_invariants_passed": candidate.validation_report.safety_invariants_passed,
                        "violations": candidate.validation_report.violations,
                        "is_promotable": candidate.validation_report.is_promotable,
                    })

                conn.execute(
                    """
                    INSERT OR REPLACE INTO candidate_updates (
                        candidate_id, component, parameter_key, current_value_json, proposed_value_json,
                        description, author, created_at, status, validation_report_json, holdout_score,
                        holdout_sample_size, stress_shortfalls, safety_passed, promoted_at, promoted_by,
                        promotion_role, rejection_reason, is_metadata_only
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        candidate.candidate_id,
                        candidate.component,
                        candidate.parameter_key,
                        json.dumps(candidate.current_value),
                        json.dumps(candidate.proposed_value),
                        candidate.description,
                        candidate.author,
                        candidate.created_at,
                        candidate.status.value,
                        v_report,
                        candidate.holdout_evaluation_score,
                        candidate.holdout_sample_size,
                        candidate.stress_shortfall_count,
                        1 if candidate.safety_check_passed else 0,
                        candidate.promoted_at,
                        candidate.promoted_by,
                        candidate.promotion_role,
                        candidate.rejection_reason,
                        1 if candidate.is_metadata_only else 0,
                    ),
                )
                conn.commit()
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to save candidate {candidate.candidate_id}: {e}") from e

    def get_candidate(self, candidate_id: str) -> Optional[CandidateUpdate]:
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(
                    "SELECT * FROM candidate_updates WHERE candidate_id = ?;", (candidate_id,)
                )
                row = cursor.fetchone()
                if not row:
                    return None
                
                v_rep = None
                if row["validation_report_json"]:
                    vd = json.loads(row["validation_report_json"])
                    v_rep = CandidateValidationReport(
                        candidate_id=vd["candidate_id"],
                        component=vd["component"],
                        validation_dataset=vd["validation_dataset"],
                        validation_sample_size=vd["validation_sample_size"],
                        champion_metric=vd["champion_metric"],
                        candidate_metric=vd["candidate_metric"],
                        metric_name=vd["metric_name"],
                        metric_improved=vd["metric_improved"],
                        stress_shortfall_count=vd["stress_shortfall_count"],
                        safety_invariants_passed=vd["safety_invariants_passed"],
                        violations=vd["violations"],
                        is_promotable=vd["is_promotable"],
                    )

                return CandidateUpdate(
                    candidate_id=row["candidate_id"],
                    component=row["component"],
                    parameter_key=row["parameter_key"],
                    current_value=json.loads(row["current_value_json"]) if row["current_value_json"] else None,
                    proposed_value=json.loads(row["proposed_value_json"]) if row["proposed_value_json"] else None,
                    description=row["description"],
                    author=row["author"],
                    created_at=row["created_at"],
                    status=CandidateStatus(row["status"]),
                    validation_report=v_rep,
                    holdout_evaluation_score=row["holdout_score"] or 0.0,
                    holdout_sample_size=row["holdout_sample_size"] or 0,
                    stress_shortfall_count=row["stress_shortfalls"] or 0,
                    safety_check_passed=bool(row["safety_passed"]),
                    promoted_at=row["promoted_at"],
                    promoted_by=row["promoted_by"],
                    promotion_role=row["promotion_role"],
                    rejection_reason=row["rejection_reason"],
                    is_metadata_only=bool(row["is_metadata_only"]),
                )
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to load candidate {candidate_id}: {e}") from e

    def get_all_candidates(self) -> List[CandidateUpdate]:
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT candidate_id FROM candidate_updates ORDER BY created_at ASC;")
                return [self.get_candidate(row["candidate_id"]) for row in cursor.fetchall()]  # type: ignore
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to query candidates: {e}") from e

    def save_version_audit(self, entry: Dict[str, Any]) -> None:
        try:
            with self._get_connection() as conn:
                conn.execute(
                    """
                    INSERT INTO version_audits (
                        entry_id, component, previous_version, new_version, timestamp,
                        author, approver, approver_role, action_type, justification
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
                    """,
                    (
                        entry["entry_id"],
                        entry["component"],
                        entry["previous_version"],
                        entry["new_version"],
                        entry["timestamp"],
                        entry["author"],
                        entry["approver"],
                        entry["approver_role"],
                        entry["action_type"],
                        entry["justification"],
                    ),
                )
                conn.commit()
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to record version audit {entry.get('entry_id')}: {e}") from e

    def get_version_audits(self) -> List[Dict[str, Any]]:
        try:
            with self._get_connection() as conn:
                cursor = conn.execute("SELECT * FROM version_audits ORDER BY timestamp ASC;")
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to query version audits: {e}") from e

    def clear(self) -> None:
        try:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM learning_events;")
                conn.execute("DELETE FROM candidate_updates;")
                conn.execute("DELETE FROM version_audits;")
                conn.commit()
        except sqlite3.Error as e:
            raise PersistenceError(f"Failed to clear database: {e}") from e

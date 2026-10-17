"""
System-Wide Self-Learning Package for Reserve Juggler.
"""

from reserve_juggler.learning.service import LearningService
from reserve_juggler.learning.models import (
    CandidateStatus,
    CandidateUpdate,
    CandidateValidationReport,
    ForecastPerformanceMetrics,
    DecisionPerformanceMetrics,
    SentinelPerformanceMetrics,
    SystemPerformanceReport,
    MatchedDecisionOutcome,
    MatchedForecastOutcome,
    MatchedOutcomePair,
    DriftAnalysisResult,
)
from reserve_juggler.learning.store import EventStore
from reserve_juggler.learning.sqlite_store import SqliteEventStore, PersistenceError
from reserve_juggler.learning.matcher import OutcomeMatcher
from reserve_juggler.learning.metrics import SystemPerformanceEvaluator
from reserve_juggler.learning.governance import SafetyInvariantValidator, VersionRegistry
from reserve_juggler.learning.runtime_registry import RuntimeComponentRegistry
from reserve_juggler.learning.evaluation_engine import CandidateEvaluationEngine

__all__ = [
    "LearningService",
    "CandidateStatus",
    "CandidateUpdate",
    "CandidateValidationReport",
    "ForecastPerformanceMetrics",
    "DecisionPerformanceMetrics",
    "SentinelPerformanceMetrics",
    "SystemPerformanceReport",
    "MatchedDecisionOutcome",
    "MatchedForecastOutcome",
    "MatchedOutcomePair",
    "DriftAnalysisResult",
    "EventStore",
    "SqliteEventStore",
    "PersistenceError",
    "OutcomeMatcher",
    "SystemPerformanceEvaluator",
    "SafetyInvariantValidator",
    "VersionRegistry",
    "RuntimeComponentRegistry",
    "CandidateEvaluationEngine",
]

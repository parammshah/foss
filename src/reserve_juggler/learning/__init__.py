"""
System-Wide Self-Learning Package for Reserve Juggler.
"""

from reserve_juggler.learning.service import LearningService
from reserve_juggler.learning.models import (
    CandidateStatus,
    CandidateUpdate,
    ForecastPerformanceMetrics,
    DecisionPerformanceMetrics,
    SentinelPerformanceMetrics,
    SystemPerformanceReport,
    MatchedOutcomePair,
)
from reserve_juggler.learning.store import EventStore
from reserve_juggler.learning.matcher import OutcomeMatcher
from reserve_juggler.learning.metrics import SystemPerformanceEvaluator
from reserve_juggler.learning.governance import SafetyInvariantValidator, VersionRegistry

__all__ = [
    "LearningService",
    "CandidateStatus",
    "CandidateUpdate",
    "ForecastPerformanceMetrics",
    "DecisionPerformanceMetrics",
    "SentinelPerformanceMetrics",
    "SystemPerformanceReport",
    "MatchedOutcomePair",
    "EventStore",
    "OutcomeMatcher",
    "SystemPerformanceEvaluator",
    "SafetyInvariantValidator",
    "VersionRegistry",
]

"""
Data models for the Macro-Panic Sentinel and Financial-Event Monitoring.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from reserve_juggler.integration.contracts import RiskSignal, EventCategory, RiskSeverity


@dataclass
class RawEventRecord:
    event_id: str
    source: str
    title: str
    summary: str
    publication_time: str
    ingestion_time: str
    category: str
    affected_markets: List[str]
    evidence_url: Optional[str] = None
    source_credibility: float = 0.8  # 0.0 - 1.0 (e.g. Fed/ECB=0.98, Bloomberg=0.88, Social=0.35)
    sentiment_score: float = 0.0  # -1.0 to 1.0
    raw_payload: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EventCluster:
    cluster_id: str
    canonical_event: RawEventRecord
    duplicates: List[RawEventRecord] = field(default_factory=list)
    first_seen: str = ""
    last_seen: str = ""
    reporting_count: int = 1
    reporting_acceleration: float = 0.0  # Events per hour acceleration


@dataclass
class SentinelEvaluationResult:
    as_of_timestamp: str
    active_signals: List[RiskSignal]
    total_events_processed: int
    duplicates_filtered: int
    aggregate_macro_stress_score: float  # 0.0 - 100.0 (0=calm, 100=extreme crisis)
    recommended_stress_scenarios: List[str]
    mode: str  # "LIVE" or "SYNTHETIC_LAB"
    system_status: str  # "OK", "DEGRADED_OFFLINE", "ERROR"
    warnings: List[str] = field(default_factory=list)

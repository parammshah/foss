"""
Macro-Panic Sentinel Service.
External financial-event intelligence and real-time stress signal evaluation.

Interface:
    MacroPanicSentinel.evaluate(as_of_timestamp: str, treasury_state: Optional[TreasuryState]) -> SentinelEvaluationResult
Version: 1.0.0
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone
from typing import Dict, List, Optional

from reserve_juggler.events.deduplication import EventDeduplicator
from reserve_juggler.events.interpreter import EventRiskInterpreter
from reserve_juggler.events.models import RawEventRecord, SentinelEvaluationResult
from reserve_juggler.events.sources import EventSource, EventSourceManager, SyntheticEventSource
from reserve_juggler.integration.contracts import RiskSeverity, RiskSignal, TreasuryState

logger = logging.getLogger(__name__)


class MacroPanicSentinel:
    """
    Macro-Panic Sentinel:
    Monitors macroeconomic announcements, rate policy decisions, banking system stress,
    repo/bond dislocations, and withdrawal accelerations. Translates raw coverage into
    calibrated RiskSignal records with relevance to actual treasury exposures.
    """

    SERVICE_VERSION = "1.0.0"

    def __init__(
        self,
        use_synthetic_mode: bool = True,
        similarity_threshold: float = 0.55,
        critical_threshold: float = 0.85,
    ):
        self.use_synthetic_mode = use_synthetic_mode
        self.source_manager = EventSourceManager(use_synthetic=use_synthetic_mode)
        self.deduplicator = EventDeduplicator(similarity_threshold=similarity_threshold)
        self.interpreter = EventRiskInterpreter(critical_severity_threshold=critical_threshold)
        self._history: List[SentinelEvaluationResult] = []

    def set_synthetic_mode(self, enabled: bool) -> None:
        """Toggle explicitly labeled synthetic-event mode for test / offline usage."""
        self.use_synthetic_mode = enabled
        self.source_manager.set_mode(enabled)

    def set_synthetic_scenario(self, scenario_name: str) -> None:
        """Switch active synthetic scenario (e.g. NORMAL, BANKING_STRESS, RATE_HIKE_SHOCK)."""
        self.source_manager.synthetic_source.set_scenario(scenario_name)

    def inject_synthetic_event(self, event: RawEventRecord) -> None:
        """Inject a custom synthetic event into the event stream."""
        self.source_manager.synthetic_source.inject_event(event)

    def add_event_source(self, source: EventSource) -> None:
        """Register an authorized external feed source (e.g. RSS/API)."""
        self.source_manager.add_live_source(source)

    def evaluate(
        self, as_of_timestamp: Optional[str] = None, treasury_state: Optional[TreasuryState] = None
    ) -> SentinelEvaluationResult:
        """
        Evaluate external financial intelligence as of a given timestamp against treasury state.

        Args:
            as_of_timestamp: ISO 8601 evaluation timestamp (defaults to UTC now).
            treasury_state: Current TreasuryState to compute exposure relevance.

        Returns:
            SentinelEvaluationResult containing active signals, stress score, and recommended scenarios.
        """
        now_iso = as_of_timestamp or datetime.now(timezone.utc).isoformat()
        warnings: List[str] = []

        # 1. Ingest raw events
        try:
            raw_events = self.source_manager.fetch_all_events()
        except Exception as e:
            logger.error("Event ingestion failed: %s. Using emergency fallback.", str(e))
            raw_events = []
            warnings.append(f"Primary feed ingestion failed: {str(e)}")

        # 2. Deduplicate and cluster
        self.deduplicator.reset()
        clusters = self.deduplicator.cluster_events(raw_events)
        total_duplicates = sum(len(c.duplicates) for c in clusters)

        # 3. Interpret risk signals
        active_signals: List[RiskSignal] = []
        recommended_scenarios: set[str] = set()
        severity_points = 0.0

        for cluster in clusters:
            signal = self.interpreter.interpret_cluster(cluster, treasury_state=treasury_state)
            active_signals.append(signal)

            # Map signal to stress scenarios
            for tag in signal.scenario_tags:
                recommended_scenarios.add(tag)

            # Score contribution
            weight = {"LOW": 5.0, "MEDIUM": 15.0, "HIGH": 30.0, "CRITICAL": 50.0}.get(signal.severity, 5.0)
            severity_points += weight * signal.confidence * signal.relevance_to_treasury

        # 4. Aggregate Macro Stress Score (0 to 100)
        macro_stress_score = min(100.0, round(severity_points, 1))

        # Default scenario recommendations if severe stress detected
        if macro_stress_score >= 60.0 and not recommended_scenarios:
            recommended_scenarios.add("COMBINED_LIQUIDITY_SQUEEZE")

        mode_str = "SYNTHETIC_LAB" if self.use_synthetic_mode else "LIVE"
        status_str = "OK" if not warnings else "DEGRADED_OFFLINE"

        result = SentinelEvaluationResult(
            as_of_timestamp=now_iso,
            active_signals=active_signals,
            total_events_processed=len(raw_events),
            duplicates_filtered=total_duplicates,
            aggregate_macro_stress_score=macro_stress_score,
            recommended_stress_scenarios=sorted(list(recommended_scenarios)),
            mode=mode_str,
            system_status=status_str,
            warnings=warnings,
        )

        self._history.append(result)
        return result

    def get_latest_result(self) -> Optional[SentinelEvaluationResult]:
        return self._history[-1] if self._history else None

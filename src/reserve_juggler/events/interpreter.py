"""
Event-Risk Interpretation Engine.
Separates raw news sentiment from verified financial events and measurable market risk.
Computes portfolio exposure relevance, corroboration status, confidence, and generates
RiskSignal records consumable by Team 2's safety score, reserve policy, and stress lab.
"""

from __future__ import annotations
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional
from reserve_juggler.events.models import EventCluster, RawEventRecord
from reserve_juggler.integration.contracts import (
    EventCategory, RiskSeverity, RiskSignal, TreasuryState
)


class EventRiskInterpreter:
    """
    Translates deduplicated event clusters into calibrated RiskSignals.
    """

    def __init__(
        self,
        high_severity_threshold: float = 0.60,
        critical_severity_threshold: float = 0.80,
        stale_hours_threshold: float = 48.0,
    ):
        self.high_severity_threshold = high_severity_threshold
        self.critical_severity_threshold = critical_severity_threshold
        self.stale_hours_threshold = stale_hours_threshold

    def calculate_treasury_relevance(
        self, event: RawEventRecord, treasury_state: Optional[TreasuryState]
    ) -> float:
        """
        Calculates relevance (0.0 to 1.0) of an event to the actual portfolio holdings.
        """
        if not treasury_state:
            return 0.5  # Neutral default if state is unknown

        asset_classes = {p.asset_class.upper() for p in treasury_state.asset_positions}
        total_assets = sum(p.market_value for p in treasury_state.asset_positions) + treasury_state.available_cash
        
        relevance_score = 0.1  # Baseline minimal relevance for any financial news

        # Check market mapping
        affected_set = {m.upper() for m in event.affected_markets}
        
        # Cash / Deposits
        if "CASH" in affected_set or "DEPOSITS" in affected_set or "WITHDRAWALS" in affected_set:
            cash_ratio = treasury_state.available_cash / total_assets if total_assets > 0 else 0.5
            relevance_score = max(relevance_score, 0.4 + 0.6 * cash_ratio)

        # Sovereign bonds / Rates
        if ("SOVEREIGN_BONDS" in affected_set or "RATES" in affected_set) and "SOVEREIGN_BONDS" in asset_classes:
            sov_value = sum(p.market_value for p in treasury_state.asset_positions if p.asset_class == "SOVEREIGN_BONDS")
            weight = sov_value / total_assets if total_assets > 0 else 0.0
            relevance_score = max(relevance_score, 0.5 + 0.5 * weight)

        # Repo
        if ("REPO" in affected_set or "INTERBANK" in affected_set) and "REPO" in asset_classes:
            repo_value = sum(p.market_value for p in treasury_state.asset_positions if p.asset_class == "REPO")
            weight = repo_value / total_assets if total_assets > 0 else 0.0
            relevance_score = max(relevance_score, 0.5 + 0.5 * weight)

        # Equities
        if ("EQUITIES" in affected_set or "STOCKS" in affected_set) and "EQUITIES" in asset_classes:
            eq_value = sum(p.market_value for p in treasury_state.asset_positions if p.asset_class == "EQUITIES")
            weight = eq_value / total_assets if total_assets > 0 else 0.0
            relevance_score = max(relevance_score, 0.3 + 0.7 * weight)

        # Banking system stress is universally relevant to any treasury
        if event.category == EventCategory.BANKING_STRESS.value:
            relevance_score = max(relevance_score, 0.85)

        return min(1.0, round(relevance_score, 3))

    def evaluate_corroboration(self, cluster: EventCluster) -> str:
        """Determines corroboration status: OFFICIAL, CORROBORATED, or UNVERIFIED."""
        canonical = cluster.canonical_event
        if canonical.source_credibility >= 0.95:
            return "OFFICIAL"
        
        # Count unique sources in cluster
        sources = {canonical.source} | {d.source for d in cluster.duplicates}
        if len(sources) >= 2 or cluster.reporting_count >= 3:
            return "CORROBORATED"
        
        return "UNVERIFIED"

    def calculate_confidence(self, cluster: EventCluster, corroboration_status: str) -> float:
        """
        Calculates confidence score (0.0 to 1.0) taking into account source credibility,
        corroboration, and staleness.
        """
        canonical = cluster.canonical_event
        base_confidence = canonical.source_credibility

        # Boost for corroboration
        if corroboration_status == "OFFICIAL":
            base_confidence = max(base_confidence, 0.95)
        elif corroboration_status == "CORROBORATED":
            base_confidence = min(1.0, base_confidence + 0.15)
        else:
            base_confidence = min(0.65, base_confidence)

        # Staleness check
        try:
            from reserve_juggler.events.sources import parse_rfc_or_iso_datetime
            now = datetime.now(timezone.utc)
            pub_iso = parse_rfc_or_iso_datetime(canonical.publication_time)
            pub_dt = datetime.fromisoformat(pub_iso.replace("Z", "+00:00"))
            age_hours = (now - pub_dt).total_seconds() / 3600.0
            if age_hours > self.stale_hours_threshold:
                # Decay confidence for stale reports
                decay = math.exp(-0.05 * (age_hours - self.stale_hours_threshold))
                base_confidence *= max(0.2, decay)
        except Exception:
            pass

        return round(min(1.0, max(0.05, base_confidence)), 3)

    def determine_severity_level(self, severity_score: float) -> str:
        if severity_score >= self.critical_severity_threshold:
            return RiskSeverity.CRITICAL.value
        elif severity_score >= self.high_severity_threshold:
            return RiskSeverity.HIGH.value
        elif severity_score >= 0.40:
            return RiskSeverity.MEDIUM.value
        return RiskSeverity.LOW.value

    def interpret_cluster(
        self, cluster: EventCluster, treasury_state: Optional[TreasuryState] = None
    ) -> RiskSignal:
        canonical = cluster.canonical_event
        relevance = self.calculate_treasury_relevance(canonical, treasury_state)
        corroboration = self.evaluate_corroboration(cluster)
        confidence = self.calculate_confidence(cluster, corroboration)

        # Category base risk
        category_weights = {
            EventCategory.BANKING_STRESS.value: 0.85,
            EventCategory.WITHDRAWAL_ACCELERATION.value: 0.90,
            EventCategory.MARKET_SHOCK.value: 0.70,
            EventCategory.MONETARY_POLICY.value: 0.60,
            EventCategory.MARKET_DISRUPTION.value: 0.65,
            EventCategory.GEO_POLITICAL.value: 0.50,
            EventCategory.GENERAL_NEWS.value: 0.20,
        }
        base_cat_weight = category_weights.get(canonical.category, 0.40)

        # Critical: Separate Sentiment from Risk
        # Sentiment indicates market tone, but severity requires tangible event impact
        sentiment_penalty = max(0.0, -canonical.sentiment_score * 0.20)
        
        # Raw severity
        raw_severity = (base_cat_weight * 0.55) + (sentiment_penalty * 0.25) + (relevance * 0.20)
        
        # Dampen if unverified
        if corroboration == "UNVERIFIED":
            raw_severity *= 0.70  # Prevent unverified rumors from causing critical panic

        # Map to scenario tags
        scenario_tags = []
        if canonical.category == EventCategory.MONETARY_POLICY.value:
            scenario_tags.append("RATES_UP_50BP")
        if canonical.category in (EventCategory.BANKING_STRESS.value, EventCategory.WITHDRAWAL_ACCELERATION.value):
            scenario_tags.extend(["DEPOSIT_RUN_SIM", "REPO_SQUEEZE_100BP"])
        if canonical.category == EventCategory.MARKET_SHOCK.value:
            scenario_tags.append("BOND_YIELD_SPIKE_40BP")

        severity_level = self.determine_severity_level(raw_severity)
        now_iso = datetime.now(timezone.utc).isoformat()

        evidence_text = (
            f"Title: {canonical.title}. Summary: {canonical.summary} "
            f"[Corroboration: {corroboration}, Mentions: {cluster.reporting_count}, "
            f"Relevance: {relevance:.0%}, Sentiment: {canonical.sentiment_score:+.2f}]"
        )

        return RiskSignal(
            signal_id=f"SIG-{cluster.cluster_id}",
            source=canonical.source,
            category=canonical.category,
            publication_time=canonical.publication_time,
            ingestion_time=now_iso,
            severity=severity_level,
            confidence=confidence,
            evidence=evidence_text,
            evidence_url=canonical.evidence_url,
            affected_markets=canonical.affected_markets,
            scenario_tags=scenario_tags,
            corroboration_status=corroboration,
            sentiment_score=canonical.sentiment_score,
            is_duplicate=False,
            relevance_to_treasury=relevance,
        )

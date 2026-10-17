"""
Integration tests for Macro-Panic Sentinel and Event Risk Interpretation.
Verifies feed failure fallback, deduplication, evidence integrity,
and RiskSignal schema conformity.
"""

import pytest
from datetime import datetime, timezone, timedelta
from reserve_juggler.events.sentinel import MacroPanicSentinel
from reserve_juggler.events.models import RawEventRecord
from reserve_juggler.events.deduplication import EventDeduplicator
from reserve_juggler.events.interpreter import EventRiskInterpreter
from reserve_juggler.events.sources import RSSFeedSource, SyntheticEventSource
from reserve_juggler.integration.contracts import RiskSignal, RiskSeverity, EventCategory
from reserve_juggler.integration.fixtures import create_mock_treasury_state


def test_sentinel_evaluates_synthetic_mode_successfully():
    """Verifies MacroPanicSentinel runs in synthetic mode without external network."""
    sentinel = MacroPanicSentinel(use_synthetic_mode=True)
    treasury_state = create_mock_treasury_state()

    result = sentinel.evaluate(treasury_state=treasury_state)

    assert result is not None
    assert result.mode in ("SYNTHETIC_DEMO", "SYNTHETIC_LAB")
    assert result.system_status == "OK"
    assert len(result.active_signals) > 0
    assert 0.0 <= result.aggregate_macro_stress_score <= 100.0


def test_event_deduplication_and_clustering():
    """Tests clustering of identical/syndicated financial headlines and echo coverage."""
    dedup = EventDeduplicator(similarity_threshold=0.50)
    now = datetime.now(timezone.utc).isoformat()

    ev1 = RawEventRecord(
        event_id="EV-1",
        source="REUTERS",
        title="Central Bank Unexpectedly Hikes Benchmark Interest Rate by 50 bps",
        summary="Monetary policy committee votes to raise policy rates amid stubborn inflation.",
        publication_time=now,
        ingestion_time=now,
        category=EventCategory.MONETARY_POLICY.value,
        affected_markets=["RATES", "SOVEREIGN_BONDS"],
        source_credibility=0.92,
    )
    # Echo report from different publisher with similar text
    ev2 = RawEventRecord(
        event_id="EV-2",
        source="BLOOMBERG",
        title="Central Bank Hikes Interest Rate by 50 bps Unexpectedly",
        summary="Policy rate increases by 50 basis points to fight inflation.",
        publication_time=now,
        ingestion_time=now,
        category=EventCategory.MONETARY_POLICY.value,
        affected_markets=["RATES", "SOVEREIGN_BONDS"],
        source_credibility=0.90,
    )
    # Distinct report
    ev3 = RawEventRecord(
        event_id="EV-3",
        source="WSJ",
        title="Oil Shipping Freight Rates Ease Following Port Re-opening",
        summary="Maritime shipping traffic resumes normal scheduling.",
        publication_time=now,
        ingestion_time=now,
        category=EventCategory.GENERAL_NEWS.value,
        affected_markets=["COMMODITIES"],
        source_credibility=0.88,
    )

    clusters = dedup.cluster_events([ev1, ev2, ev3])
    
    # ev1 and ev2 should be clustered together into 1 cluster
    assert len(clusters) == 2
    rate_cluster = next(c for c in clusters if "Hikes" in c.canonical_event.title or "Interest" in c.canonical_event.title)
    assert rate_cluster.reporting_count == 2
    assert len(rate_cluster.duplicates) == 1


def test_event_risk_interpretation_schema_and_relevance():
    """Verifies RiskSignal fields, confidence decay on stale reports, and treasury relevance calculation."""
    interpreter = EventRiskInterpreter(stale_hours_threshold=24.0)
    treasury_state = create_mock_treasury_state()

    # Create cluster for banking stress
    now = datetime.now(timezone.utc).isoformat()
    ev = RawEventRecord(
        event_id="EV-STRESS",
        source="CENTRAL_BANK",
        title="Interbank Funding Stress Prompts Emergency Liquidity Facility",
        summary="Regulators open discount window as credit spreads widen.",
        publication_time=now,
        ingestion_time=now,
        category=EventCategory.BANKING_STRESS.value,
        affected_markets=["CASH", "REPO", "DEPOSITS"],
        evidence_url="https://centralbank.gov/notice",
        source_credibility=0.98,
        sentiment_score=-0.80,
    )
    
    dedup = EventDeduplicator()
    clusters = dedup.cluster_events([ev])
    signal = interpreter.interpret_cluster(clusters[0], treasury_state=treasury_state)

    # Verify RiskSignal contract
    assert isinstance(signal, RiskSignal)
    assert signal.signal_id.startswith("SIG-")
    assert signal.severity in ("HIGH", "CRITICAL")
    assert signal.corroboration_status == "OFFICIAL"
    assert signal.confidence >= 0.90
    assert signal.relevance_to_treasury > 0.80
    assert "DEPOSIT_RUN_SIM" in signal.scenario_tags


def test_stale_event_confidence_decay():
    """Verifies that stale reports (older than 48 hours) suffer confidence decay."""
    interpreter = EventRiskInterpreter(stale_hours_threshold=48.0)
    stale_time = (datetime.now(timezone.utc) - timedelta(hours=96)).isoformat()
    
    ev_stale = RawEventRecord(
        event_id="EV-OLD",
        source="UNOFFICIAL_BLOG",
        title="Rumor of Regional Bank Depositor Outflows",
        summary="Anonymous sources claim elevated withdrawals.",
        publication_time=stale_time,
        ingestion_time=datetime.now(timezone.utc).isoformat(),
        category=EventCategory.WITHDRAWAL_ACCELERATION.value,
        affected_markets=["CASH"],
        source_credibility=0.60,
    )
    
    dedup = EventDeduplicator()
    clusters = dedup.cluster_events([ev_stale])
    signal = interpreter.interpret_cluster(clusters[0])

    assert signal.confidence < 0.40  # Stale + unverified decay


def test_rss_feed_failure_resilience():
    """Verifies that network failure in RSS source does not crash system and degrades gracefully."""
    bad_source = RSSFeedSource(
        name="UNREACHABLE_FEED",
        feed_url="https://nonexistent-treasury-domain-12345.org/rss.xml",
        timeout_seconds=0.5,
    )
    events = bad_source.fetch_events()
    assert events == []  # Gracefully returns empty, does not raise exception
    assert bad_source.status == "UNAVAILABLE"
    assert bad_source.last_error is not None


def test_rfc2822_date_parsing_and_since_filtering():
    """Verifies RFC 2822 RSS pubDate parsing, timezone normalization, and since_iso filtering."""
    from reserve_juggler.events.sources import parse_rfc_or_iso_datetime

    # Valid RFC 2822 date
    rfc_date = "Wed, 02 Oct 2026 13:00:00 GMT"
    parsed_utc = parse_rfc_or_iso_datetime(rfc_date)
    assert parsed_utc is not None
    assert "2026-10-02T13:00:00" in parsed_utc
    assert "+00:00" in parsed_utc or "Z" in parsed_utc

    # Malformed date must return None to prevent stale/invalid events from appearing fresh
    bad_date = "invalid-date-string-xyz"
    parsed_bad = parse_rfc_or_iso_datetime(bad_date)
    assert parsed_bad is None  # Does not substitute now_utc


def test_financial_concept_classification():
    """Verifies deterministic concept classification, sentiment scoring, and market tagging."""
    from reserve_juggler.events.classifier import FinancialEventClassifier

    classifier = FinancialEventClassifier()

    # Rate hike headline
    res_rate = classifier.classify(
        title="Federal Reserve Unexpectedly Raises Benchmark Policy Rate by 75 bps",
        summary="FOMC acts aggressively against persistent core inflation pressures.",
    )
    assert res_rate["category"] == "MONETARY_POLICY"
    assert res_rate["sentiment_score"] < -0.2
    assert "RATES" in res_rate["affected_markets"]
    assert "SOVEREIGN_BONDS" in res_rate["affected_markets"]

    # Bank failure / run headline
    res_bank = classifier.classify(
        title="Regulators Take Possession of Troubled Regional Bank Following Massive Deposit Run",
        summary="Emergency weekend resolution after depositors withdrew $42 billion in 24 hours.",
    )
    assert res_bank["category"] == "BANKING_STRESS"
    assert res_bank["severity"] == "CRITICAL"
    assert res_bank["sentiment_score"] < -0.6
    assert "CASH" in res_bank["affected_markets"]


def test_stateful_deduplication_velocity_across_multiple_evaluations():
    """
    Verifies that EventDeduplicator persists history across evaluations,
    calculates reporting velocity, and does not reset between calls.
    """
    dedup = EventDeduplicator()
    now_dt = datetime.now(timezone.utc)
    now = now_dt.isoformat()

    ev = RawEventRecord(
        event_id="EV-INITIAL",
        source="REUTERS",
        title="Sovereign Bond Yields Spike 40 bps Across 10-Year Benchmark Curve",
        summary="Global fixed income rout deepens on higher inflation forecasts.",
        publication_time=now,
        ingestion_time=now,
        category="MARKET_SHOCK",
        affected_markets=["SOVEREIGN_BONDS"],
    )

    # First evaluation
    clusters_1 = dedup.cluster_events([ev])
    assert len(clusters_1) == 1
    assert clusters_1[0].reporting_velocity_per_hour == 1.0

    # Repeat syndicated wire 10 minutes later (in same deduplicator instance)
    ten_min_later = (now_dt + timedelta(minutes=10)).isoformat()
    ev_repeat = RawEventRecord(
        event_id="EV-WIRE-COPY",
        source="SYNDICATED_WIRE",
        title="Sovereign Bond Yields Spike 40 bps on Fixed Income Curve",
        summary="Global bond selloff escalates as yields touch multi-year highs.",
        publication_time=ten_min_later,
        ingestion_time=ten_min_later,
        category="MARKET_SHOCK",
        affected_markets=["SOVEREIGN_BONDS"],
    )

    clusters_2 = dedup.cluster_events([ev_repeat])
    # The duplicate is recognized and clustered with prior history
    assert len(clusters_2) == 1
    assert clusters_2[0].reporting_count == 2
    # Velocity increases to > 1.0/hr because 2 events arrived in a 10-minute span
    assert clusters_2[0].reporting_velocity_per_hour > 1.0


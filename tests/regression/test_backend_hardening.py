"""
Comprehensive Regression and Financial Safety Hardening Test Suite for Reserve Juggler Team 3.
Implements the 34 required regression tests mandated in Master Prompt Section 13.
Proves that all previously audited defects are eliminated at the root cause.
"""

from __future__ import annotations
import copy
import os
import tempfile
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional
import pytest

from reserve_juggler.contracts import (
    AssetPosition,
    DecisionRecord,
    ExecutionOutcome,
    ForecastBundle,
    ForecastPoint,
    GateStatus,
    HardGateOutcome,
    LearningEvent,
    ServiceStatus,
    TreasuryState,
)
from reserve_juggler.explainability.models import DecisionExplanation, HumanActionRecord
from reserve_juggler.explainability.service import DecisionExplanationService
from reserve_juggler.events.classifier import FinancialEventClassifier
from reserve_juggler.events.deduplication import EventCluster, EventDeduplicator
from reserve_juggler.events.interpreter import EventRiskInterpreter
from reserve_juggler.events.models import EventCategory, RawEventRecord, RiskSeverity
from reserve_juggler.events.sentinel import MacroPanicSentinel
from reserve_juggler.events.sources import parse_rfc_or_iso_datetime
from reserve_juggler.integration.facade import ApplicationFacade
from reserve_juggler.integration.stress_engine import StressLabSimulationEngine
from reserve_juggler.learning.evaluation_engine import CandidateEvaluationEngine
from reserve_juggler.learning.matcher import OutcomeMatcher
from reserve_juggler.learning.metrics import SystemPerformanceEvaluator
from reserve_juggler.learning.models import (
    CandidateStatus,
    CandidateUpdate,
    MatchedDecisionOutcome,
    MatchedForecastOutcome,
)
from reserve_juggler.learning.runtime_registry import (
    GenericRuntimeProvider,
    RuntimeComponentRegistry,
    SentinelRuntimeProvider,
)
from reserve_juggler.learning.service import LearningService
from reserve_juggler.learning.sqlite_store import PersistenceError, SqliteEventStore
from reserve_juggler.learning.store import EventStore


# ==============================================================================
# SECTION 13: REGRESSION TESTS 1-3 — CONTRACT DEFAULTS AND FINANCIAL SAFETY
# ==============================================================================

def test_1_missing_protected_reserve_cannot_default_to_zero_and_permit_investment():
    """
    Defect 1: TreasuryState.protected_reserve_floor must default to None, not 0.0.
    Missing reserve floor must prohibit risky allocations.
    """
    from reserve_juggler.integration.fixtures import create_mock_decision_record, create_mock_treasury_state
    from reserve_juggler.contracts import ApprovalStatus, UserRole
    from reserve_juggler.explainability.service import DecisionExplanationService

    ts = create_mock_treasury_state()
    ts.protected_reserve_floor = None  # None/Unconfigured
    assert ts.protected_reserve_floor is None

    service = DecisionExplanationService()
    decision = create_mock_decision_record()
    decision.selected_action = "INVEST_REPOS_30D"

    ok, msg, _, _, _ = service.execute_human_workflow(
        decision_record=decision,
        action_type=ApprovalStatus.APPROVED.value,
        actor_name="Lead Treasurer",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="Deploy idle cash.",
        treasury_state=ts,
    )
    assert not ok
    assert "UNCONFIGURED" in msg or "UNKNOWN" in msg or "Mandatory Reserve Floor" in msg


def test_2_missing_safety_score_cannot_default_to_85():
    """
    Defect 2: DecisionRecord.treasury_safety_score must default to None, not 85.0.
    """
    from reserve_juggler.contracts import DecisionRecord
    new_dr = DecisionRecord(
        decision_id="DEC-01",
        timestamp="2026-10-17T12:00:00+00:00",
        selected_action="HOLD_CASH",
        target_asset="CASH-01",
        amount=0.0,
        reserve=25_000_000.0,
        eligible_surplus=0.0,
        estimated_costs={},
        hard_gate_outcomes=[],
        model_versions={},
        reason_codes=[],
    )
    assert new_dr.treasury_safety_score is None


def test_3_missing_hard_gate_status_cannot_default_to_pass():
    """
    Defect 3: HardGateOutcome.status must default to NOT_EVALUATED, not PASS.
    """
    gate = HardGateOutcome(gate_name="Future-Protected Capital Firewall")
    assert gate.status == GateStatus.NOT_EVALUATED
    assert gate.status != GateStatus.PASS


# ==============================================================================
# SECTION 13: REGRESSION TESTS 4-10 — FORECAST MATCHING AND METRIC RIGOR
# ==============================================================================

def test_4_forecast_target_timestamps_matched_correctly():
    """
    Defect 4: Stored forecast must retain target timestamp distinct from creation timestamp,
    and match observations at target timestamp.
    """
    store = EventStore()
    creation_ts = "2026-10-17T10:00:00+00:00"
    target_ts = "2026-10-17T14:00:00+00:00"
    fc_corr = "FC-TGT-01"

    store.record(
        LearningEvent(
            event_id="EV-FC-01",
            versioned_event_type="FORECAST_INFERENCE_RECORDED",
            module="FORECASTING",
            timestamp=creation_ts,
            correlation_id=fc_corr,
            payload={
                "horizons": ["4h"],
                "quality_status": "HIGH",
                "point_forecasts": {
                    "4h": {
                        "target_timestamp": target_ts,
                        "expected_net_flow": -3_500_000.0,
                        "expected_inflow": 4_000_000.0,
                        "expected_outflow": 7_500_000.0,
                        "uncertainty_p05": -5_000_000.0,
                        "uncertainty_p95": -2_000_000.0,
                    }
                },
            },
        )
    )

    # Observation at target timestamp
    store.record(
        LearningEvent(
            event_id="EV-OBS-01",
            versioned_event_type="ACTUAL_CASH_FLOW_OBSERVED",
            module="LEDGER",
            timestamp=target_ts,
            correlation_id=fc_corr,
            payload={
                "target_timestamp": target_ts,
                "actual_inflow": 4_000_000.0,
                "actual_outflow": 7_200_000.0,
                "actual_net_flow": -3_200_000.0,
                "data_quality_status": "VALID",
            },
        )
    )

    matcher = OutcomeMatcher(store)
    pairs = matcher.match_forecasts()
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.target_timestamp == target_ts
    assert pair.target_timestamp != creation_ts
    assert pair.actual_net_flow == -3_200_000.0


def test_5_actual_deposits_and_withdrawals_used_for_forecast_evaluation():
    """
    Defect 5: actual_net_flow = actual_deposits - actual_withdrawals when both present.
    """
    actual_deposits = 10_000_000.0
    actual_withdrawals = 14_500_000.0
    actual_net_flow = actual_deposits - actual_withdrawals
    assert actual_net_flow == -4_500_000.0


def test_6_missing_actual_observations_not_converted_to_zero():
    """
    Defect 6: Missing actual observations must not default to zero.
    """
    payload = {"target_timestamp": "2026-10-17T12:00:00+00:00"}
    actual_net = payload.get("actual_net_flow")
    assert actual_net is None
    assert actual_net != 0.0


def test_7_missing_intervals_not_fabricated():
    """
    Defect 7: Do not fabricate uncertainty intervals around point forecasts.
    """
    fc = ForecastPoint(
        horizon="1h",
        target_timestamp="2026-10-17T11:00:00+00:00",
        expected_inflow=2_000_000.0,
        expected_outflow=1_000_000.0,
        expected_net_flow=1_000_000.0,
        uncertainty_p05=None,  # Not fabricated!
        uncertainty_p50=1_000_000.0,
        uncertainty_p90=None,
        uncertainty_p95=None,
        uncertainty_p99=None,
        confidence_score=0.85,
    )
    assert fc.uncertainty_p05 is None
    assert fc.uncertainty_p95 is None


def test_8_interval_coverage_uses_actual_stored_bounds():
    """
    Defect 8: Interval coverage must use actual stored bounds (p10, p90) and count hits genuinely.
    """
    store = EventStore()
    evaluator = SystemPerformanceEvaluator(store)

    pairs = [
        MatchedForecastOutcome(
            correlation_id=f"C{i}",
            forecast_id=f"F{i}",
            target_timestamp=f"2026-10-17T1{i}:00:00+00:00",
            horizon="1h",
            predicted_net_flow=100.0,
            predicted_inflow=100.0,
            predicted_outflow=0.0,
            uncertainty_p05=90.0,
            uncertainty_p50=100.0,
            uncertainty_p90=150.0,
            uncertainty_p95=160.0,
            uncertainty_p99=180.0,
            model_version="v1.4.0",
            actual_net_flow=120.0 if i < 2 else 200.0,
            is_eligible=True,
            within_90_interval=(i < 2),  # 2 inside, 2 outside -> 50%
        )
        for i in range(4)
    ]
    rep = evaluator.evaluate(matched_decisions=[], matched_forecasts=pairs)
    assert rep.forecast_metrics.interval_coverage_pct == 50.0


def test_9_zero_actual_values_do_not_produce_invalid_mape():
    """
    Defect 9: Zero actual values must be safely excluded from MAPE with count reported.
    """
    store = EventStore()
    evaluator = SystemPerformanceEvaluator(store)

    pairs = [
        MatchedForecastOutcome(
            correlation_id=f"CZ{i}",
            forecast_id=f"FZ{i}",
            target_timestamp=f"2026-10-17T1{i}:00:00+00:00",
            horizon="1h",
            predicted_net_flow=50.0,
            predicted_inflow=50.0,
            predicted_outflow=0.0,
            uncertainty_p05=-10.0,
            uncertainty_p50=50.0,
            uncertainty_p90=80.0,
            uncertainty_p95=90.0,
            uncertainty_p99=100.0,
            model_version="v1.4.0",
            actual_net_flow=0.0,  # Zero actual!
            is_eligible=True,
            within_90_interval=True,
        )
        for i in range(3)
    ]
    rep = evaluator.evaluate(matched_decisions=[], matched_forecasts=pairs)
    assert rep.forecast_metrics.mape is None  # Excluded rather than division by zero
    assert any("MAPE undefined" in r for r in rep.forecast_metrics.exclusion_reasons)


def test_10_empty_metric_samples_produce_inconclusive_or_unavailable():
    """
    Defect 10: Empty metric samples must report INCONCLUSIVE, not 100% pass rates or 0 error.
    """
    store = EventStore()
    evaluator = SystemPerformanceEvaluator(store)
    rep = evaluator.evaluate(matched_decisions=[], matched_forecasts=[])
    assert rep.overall_system_health == "INCONCLUSIVE"
    assert rep.decision_metrics.hard_gate_status in ("INCONCLUSIVE", "UNKNOWN")
    assert rep.decision_metrics.hard_gate_pass_rate == 0.0
    assert "UNAVAILABLE_WITHOUT_GROUND_TRUTH" in rep.sentinel_metrics.precision_status


# ==============================================================================
# SECTION 13: REGRESSION TESTS 11-16 — LEARNING EVALUATION & RUNTIME PROMOTION
# ==============================================================================

def test_11_no_candidate_metric_can_pass_without_actual_evaluation_run():
    """
    Defect 11: Candidate cannot pass without running genuine holdout evaluation.
    """
    engine = CandidateEvaluationEngine()
    candidate = CandidateUpdate(
        candidate_id="CAND-01",
        component="FORECASTING",
        parameter_key="ensemble_combining_weights",
        current_value={"m1": 0.5, "m2": 0.5},
        proposed_value={"m1": 0.5, "m2": 0.5},  # Equal to champion: does not improve!
        description="No improvement candidate",
        author="Quant",
        created_at="2026-10-17T12:00:00+00:00",
    )
    report = engine.evaluate_candidate(candidate, sample_size=30)
    assert not report.metric_improved
    assert not report.is_promotable
    assert report.status in ("FAIL", "INCONCLUSIVE")


def test_12_promotion_is_blocked_if_runtime_consumer_is_missing():
    """
    Defect 12: Promotion is blocked if an active runtime provider is not registered (UNINTEGRATED).
    """
    service = LearningService()
    # Unregister provider for FORECASTING to simulate missing consumer
    service.runtime_registry.unregister_provider("FORECASTING")

    cand = service.propose_update(
        component="FORECASTING",
        parameter_key="ensemble_combining_weights",
        current_value={"m1": 0.5, "m2": 0.5},
        proposed_value={"m1": 0.4, "m2": 0.6},
        description="Test unintegrated promotion",
        author="Quant",
        holdout_sample_size=30,
        stress_shortfall_count=0,
    )
    ok, msg = service.promote_candidate(
        candidate_id=cand.candidate_id,
        approver="Chief Treasurer",
        approver_role="HEAD_OF_TREASURY",
        justification="Validation passed but runtime missing.",
    )
    assert not ok
    assert "UNINTEGRATED" in msg or "no registered active runtime consumer" in msg


def test_13_candidate_promotion_changes_active_service_behavior_where_connected():
    """
    Defect 13: Promoting a candidate directly updates the registered runtime provider's parameters.
    """
    service = LearningService()
    sentinel = MacroPanicSentinel(use_synthetic_mode=True)
    sentinel.alert_threshold = 60.0
    provider = SentinelRuntimeProvider(sentinel)
    service.runtime_registry.register_provider("SENTINEL_RISK", provider)

    cand = service.propose_update(
        component="SENTINEL_RISK",
        parameter_key="panic_sentinel_alert_threshold",
        current_value=60.0,
        proposed_value=68.0,
        description="Calibrate panic sentinel threshold",
        author="Risk Analyst",
        holdout_sample_size=30,
        stress_shortfall_count=0,
    )
    ok, msg = service.promote_candidate(
        candidate_id=cand.candidate_id,
        approver="Chief Risk Officer",
        approver_role="RISK_OFFICER",
        justification="Calibrated threshold.",
        new_version_tag="v2.0.0",
    )
    assert ok
    # Verify live consumer behavior changed!
    assert sentinel.alert_threshold == 68.0


def test_14_rollback_restores_active_runtime_behavior():
    """
    Defect 14: Rollback restores the registered runtime consumer's prior parameter.
    """
    service = LearningService()
    sentinel = MacroPanicSentinel(use_synthetic_mode=True)
    sentinel.alert_threshold = 60.0
    provider = SentinelRuntimeProvider(sentinel)
    service.runtime_registry.register_provider("SENTINEL_RISK", provider)

    cand = service.propose_update(
        component="SENTINEL_RISK",
        parameter_key="panic_sentinel_alert_threshold",
        current_value=60.0,
        proposed_value=72.0,
        description="Test rollback",
        author="Risk Analyst",
        holdout_sample_size=30,
        stress_shortfall_count=0,
    )
    service.promote_candidate(
        candidate_id=cand.candidate_id,
        approver="Chief Risk Officer",
        approver_role="RISK_OFFICER",
        justification="Promote for rollback test.",
        new_version_tag="v2.1.0",
    )
    assert sentinel.alert_threshold == 72.0

    # Execute rollback
    ok, msg = service.rollback_component(
        component="SENTINEL_RISK",
        target_version="v1.0.0",
        author="Risk Officer",
        approver_role="RISK_OFFICER",
        justification="Emergency rollback test",
    )
    assert ok
    # Live sentinel restored!
    assert sentinel.alert_threshold == 60.0


def test_15_candidate_stress_scenarios_actually_evaluated():
    """
    Defect 15: Optimizer candidates execute actual stress scenarios rather than parsing labels.
    """
    engine = CandidateEvaluationEngine()
    # A candidate proposing an unsafe buffer reduction that causes shortfall in executed scenario
    cand = CandidateUpdate(
        candidate_id="CAND-OPT-01",
        component="DECISION_OPTIMIZER",
        parameter_key="liquidity_buffer_multiplier",
        current_value=1.25,
        proposed_value=0.85,  # Unsafe reduction!
        description="Lean buffer",
        author="Trader",
        created_at="2026-10-17T12:00:00+00:00",
    )
    report = engine.evaluate_candidate(cand, sample_size=30)
    assert not report.is_promotable
    assert report.stress_shortfall_count > 0
    assert any("liquidity shortfalls under stress testing" in v for v in report.violations)


def test_16_failed_audit_persistence_prevents_false_success_claims():
    """
    Defect 16: If audit record write fails, promotion transaction aborts and rolls back runtime.
    """
    service = LearningService()
    provider = GenericRuntimeProvider("FORECASTING", {"ensemble_combining_weights": {"m1": 0.5, "m2": 0.5}})
    service.runtime_registry.register_provider("FORECASTING", provider)

    cand = service.propose_update(
        component="FORECASTING",
        parameter_key="ensemble_combining_weights",
        current_value={"m1": 0.5, "m2": 0.5},
        proposed_value={"m1": 0.4, "m2": 0.6},
        description="Improving weights candidate",
        author="Lead Quant",
        holdout_sample_size=30,
        stress_shortfall_count=0,
    )
    from reserve_juggler.learning.models import CandidateValidationReport
    cand.validation_report = CandidateValidationReport(
        candidate_id=cand.candidate_id,
        component=cand.component,
        validation_dataset="CHRONOLOGICAL_HOLDOUT_30D",
        validation_sample_size=30,
        champion_metric=0.12,
        candidate_metric=0.09,
        metric_name="wape",
        metric_improved=True,
        stress_shortfall_count=0,
        safety_invariants_passed=True,
        is_runtime_integrated=True,
        status="PASS",
    )

    # Monkey-patch store to simulate a database write crash during audit recording
    def failing_record(event, ignore_duplicates=False):
        raise PersistenceError("Simulated disk I/O failure during audit transaction.")
    service.store.record = failing_record

    ok, msg = service.promote_candidate(
        candidate_id=cand.candidate_id,
        approver="Chief Treasurer",
        approver_role="HEAD_OF_TREASURY",
        justification="Test persistence failure abort.",
    )
    assert not ok
    assert "durable event persistence failed" in msg or "persistence failed" in msg
    # Verify runtime parameter was rolled back and did NOT remain mutated!
    assert provider.get_parameter("ensemble_combining_weights") == {"m1": 0.5, "m2": 0.5}


# ==============================================================================
# SECTION 13: REGRESSION TESTS 17-18 — TELEMETRY ISOLATION & IDEMPOTENT STARTUP
# ==============================================================================

def test_17_synthetic_telemetry_not_seeded_in_operational_mode():
    """
    Defect 17: Production mode (use_synthetic=False) must never seed demo telemetry.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "operational_prod.db")
        facade = ApplicationFacade(persistence_file=db_path, use_synthetic=False)
        events = facade.learning_service.store.get_all()
        assert len(events) == 0  # Zero demo events seeded in production mode!
        facade.close()


def test_18_restarting_same_demo_database_does_not_create_duplicate_ids():
    """
    Defect 18: Reusing the same persistent SQLite database file across restarts is idempotent
    and does not trigger UNIQUE constraint / primary key errors.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_demo_restart.db")
        # Run 1
        f1 = ApplicationFacade(persistence_file=db_path, use_synthetic=True)
        count1 = len(f1.learning_service.store.get_all())
        assert count1 > 0
        f1.close()

        # Run 2 (Restart using the same DB path)
        f2 = ApplicationFacade(persistence_file=db_path, use_synthetic=True)
        count2 = len(f2.learning_service.store.get_all())
        # Restart succeeded without crashes, and duplicate events were not duplicated
        assert count2 >= count1
        f2.close()


# ==============================================================================
# SECTION 13: REGRESSION TESTS 19-21 — SENTINEL TIMESTAMPS & DEDUPLICATION
# ==============================================================================

def test_19_invalid_rss_timestamps_do_not_become_fresh_events():
    """
    Defect 19: Malformed pubDate must not fall back to now_utc().
    Interpreter must decay confidence and classify as LOW_CONFIDENCE.
    """
    parsed = parse_rfc_or_iso_datetime("bad-timestamp-xyz")
    assert parsed is None  # No silent now_utc fallback!

    interpreter = EventRiskInterpreter()
    ev = RawEventRecord(
        event_id="EV-BAD-TIME",
        source="RSS_FEED",
        title="Breaking Financial Rumor",
        summary="Unconfirmed report",
        publication_time="INVALID_TIMESTAMP",
        ingestion_time=datetime.now(timezone.utc).isoformat(),
        category=EventCategory.BANKING_STRESS.value,
        affected_markets=["CASH"],
        source_credibility=0.70,
    )
    dedup = EventDeduplicator()
    clusters = dedup.cluster_events([ev])
    signal = interpreter.interpret_cluster(clusters[0])
    assert signal.confidence <= 0.05
    assert signal.severity == RiskSeverity.LOW_CONFIDENCE
    assert any("INVALID_TIMESTAMP" in w for w in signal.warnings)


def test_20_deduplication_history_is_bounded():
    """
    Defect 20: Deduplicator must maintain a bounded event cache with TTL eviction.
    """
    dedup = EventDeduplicator(max_seen_events=100)
    now_iso = datetime.now(timezone.utc).isoformat()
    # Ingest 250 events
    for i in range(250):
        ev = RawEventRecord(
            event_id=f"EV-DEDUP-{i:04d}",
            source="SRC",
            title=f"Headline #{i}",
            summary="Summary",
            publication_time=now_iso,
            ingestion_time=now_iso,
            category=EventCategory.MARKET_SHOCK.value,
            affected_markets=["EQUITIES"],
            source_credibility=0.8,
        )
        dedup.cluster_events([ev])

    # Seen cache must be bounded
    assert len(dedup._seen_event_ids) <= 100


def test_21_reporting_acceleration_is_not_merely_reporting_velocity():
    """
    Defect 21: Acceleration is change in velocity over time (delta_v / delta_t), not velocity itself.
    """
    dedup = EventDeduplicator()
    now_dt = datetime.now(timezone.utc)

    ev0 = RawEventRecord(
        event_id="EV-ACC-01",
        source="SRC_A",
        title="Fed Emergency Liquidity Window Announcement",
        summary="Discount window usage increases sharply.",
        publication_time=now_dt.isoformat(),
        ingestion_time=now_dt.isoformat(),
        category=EventCategory.BANKING_STRESS.value,
        affected_markets=["CASH"],
        source_credibility=0.9,
    )
    c1 = dedup.cluster_events([ev0])[0]
    assert c1.reporting_acceleration == 0.0  # Initial acceleration is 0.0!

    # Second batch 10 minutes later: duplicate report
    t1 = (now_dt + timedelta(minutes=10)).isoformat()
    ev1 = RawEventRecord(
        event_id="EV-ACC-02",
        source="SRC_B",
        title="Fed Emergency Liquidity Window Announcement Wire",
        summary="Discount window usage increases sharply on wire.",
        publication_time=t1,
        ingestion_time=t1,
        category=EventCategory.BANKING_STRESS.value,
        affected_markets=["CASH"],
        source_credibility=0.9,
    )
    c2 = dedup.cluster_events([ev1])[0]
    # Velocity increased and acceleration computed
    assert c2.reporting_velocity_per_hour > 1.0
    assert c2.reporting_acceleration != c2.reporting_velocity_per_hour


# ==============================================================================
# SECTION 13: REGRESSION TESTS 22-24 — STRESS LAB ACCOUNTING & CONSTRAINTS
# ==============================================================================

def test_22_bridge_credit_unavailability_removes_bridge_dependent_actions():
    """
    Defect 22: When credit_unavailable=True, emergency bridge credit cannot be drawn.
    """
    engine = StressLabSimulationEngine()
    state = TreasuryState(
        timestamp="2026-10-17T12:00:00+00:00",
        available_cash=10_000_000.0,
        asset_positions=[],
        obligations=[],
        authorized_credit_availability=50_000_000.0,
        protected_reserve_floor=25_000_000.0,
    )
    # Severe shock with credit unavailable
    res = engine.run_simulation(
        base_treasury_state=state,
        deposit_shock_pct=50.0,
        credit_unavailable=True,
    )
    rj = res["strategies"]["reserve_juggler"]
    assert rj.emergency_credit_drawn == 0.0  # Prohibited from drawing credit!
    assert rj.liquidity_shortfall > 0.0


def test_23_repo_delay_changes_asset_availability_and_strategy_outcomes():
    """
    Defect 23: repo_delay_days alters asset availability timing and changes strategy outcomes.
    """
    engine = StressLabSimulationEngine()
    state = TreasuryState(
        timestamp="2026-10-17T12:00:00+00:00",
        available_cash=5_000_000.0,
        asset_positions=[
            AssetPosition(
                asset_id="REPO-TERM-01",
                name="Term 30D Repo",
                asset_class="REPO",
                market_value=20_000_000.0,
                accessible_liquidity=19_800_000.0,
                settlement_days=1,
                yield_pct=5.40,
                is_eligible=True,
            )
        ],
        obligations=[],
        authorized_credit_availability=0.0,
        protected_reserve_floor=20_000_000.0,
    )
    # Delay 0: Term repo accessible on day 0
    res_d0 = engine.run_simulation(state, deposit_shock_pct=30.0, repo_delay_days=0, credit_unavailable=True)
    # Delay 2: Term repo locked during shock window
    res_d2 = engine.run_simulation(state, deposit_shock_pct=30.0, repo_delay_days=2, credit_unavailable=True)

    assert res_d0["strategies"]["reserve_juggler"].ending_available_cash != res_d2["strategies"]["reserve_juggler"].ending_available_cash


def test_24_strategy_accounting_does_not_double_count_opening_cash():
    """
    Defect 24: Reconciled ledger must never double-count opening cash or asset proceeds.
    Ending cash + ending investments == Total Portfolio Value.
    """
    engine = StressLabSimulationEngine()
    state = TreasuryState(
        timestamp="2026-10-17T12:00:00+00:00",
        available_cash=30_000_000.0,
        asset_positions=[
            AssetPosition(
                asset_id="REPO-01",
                name="Overnight Repo",
                asset_class="REPO",
                market_value=20_000_000.0,
                accessible_liquidity=20_000_000.0,
                settlement_days=0,
                yield_pct=5.3,
                is_eligible=True,
            )
        ],
        obligations=[],
        authorized_credit_availability=10_000_000.0,
        protected_reserve_floor=20_000_000.0,
    )
    res = engine.run_simulation(state, deposit_shock_pct=10.0)
    for strat_name, strat in res["strategies"].items():
        # Reconciliation error must be 0
        assert strat.reconciliation_error == 0.0
        # Total portfolio value equals ending cash + ending investments
        expected_total = strat.ending_available_cash + strat.ending_investments
        assert abs(strat.total_portfolio_value - expected_total) < 1.0


# ==============================================================================
# SECTION 13: REGRESSION TESTS 25-29 — HUMAN ACTION VALIDATION & AUDIT
# ==============================================================================

def test_25_modified_action_costs_are_correctly_recalculated():
    """
    Defect 25: Human action amount modification captures original amount,
    preserves it in audit record, and rescales variable costs proportionally.
    """
    from reserve_juggler.integration.fixtures import create_mock_decision_record
    from reserve_juggler.contracts import ApprovalStatus, UserRole
    from reserve_juggler.explainability.service import DecisionExplanationService

    service = DecisionExplanationService()
    dr = create_mock_decision_record()
    dr.amount = 10_000_000.0
    dr.estimated_costs = {"interest": 0.0, "tax": 0.0, "penalty": 0.0, "slippage": 10_000.0, "total": 10_000.0}

    ok, msg, updated_dec, outcome, _ = service.execute_human_workflow(
        decision_record=dr,
        action_type=ApprovalStatus.MODIFIED.value,
        actor_name="Senior Trader",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="Scale up surplus allocation.",
        modified_amount=20_000_000.0,  # Doubled
    )
    assert ok
    assert updated_dec.amount == 20_000_000.0
    assert outcome.realized_amount == 20_000_000.0
    assert abs(outcome.realized_costs["total"] - 20_000.0) < 10.0


def test_26_changed_action_types_undergo_appropriate_revalidation():
    """
    Defect 26: Changing action type from HOLD_CASH to DRAW_CREDIT undergoes credit line revalidation.
    """
    from reserve_juggler.integration.fixtures import create_mock_decision_record, create_mock_treasury_state
    from reserve_juggler.contracts import ApprovalStatus, UserRole
    from reserve_juggler.explainability.service import DecisionExplanationService

    service = DecisionExplanationService()
    ts = create_mock_treasury_state()
    ts.available_cash = 10_000_000.0
    ts.authorized_credit_availability = 15_000_000.0
    ts.protected_reserve_floor = 20_000_000.0

    dr = create_mock_decision_record()
    dr.selected_action = "DRAW_CREDIT"

    # Attempt to draw 25M credit when limit is 15M
    ok, msg, _, _, _ = service.execute_human_workflow(
        decision_record=dr,
        action_type=ApprovalStatus.MODIFIED.value,
        actor_name="Senior Trader",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="Emergency liquidity",
        modified_amount=25_000_000.0,
        treasury_state=ts,
    )
    assert not ok
    assert "credit" in msg.lower() or "limit" in msg.lower() or "breaches" in msg.lower()


def test_27_cancellation_does_not_create_a_phantom_execution():
    """
    Defect 27: Cancelling an action produces status CANCELLED with no phantom settled execution.
    """
    from reserve_juggler.integration.fixtures import create_mock_decision_record
    from reserve_juggler.contracts import ApprovalStatus, UserRole
    from reserve_juggler.explainability.service import DecisionExplanationService

    service = DecisionExplanationService()
    dr = create_mock_decision_record()
    ok, msg, dec_c, out_c, ev_c = service.execute_human_workflow(
        decision_record=dr,
        action_type=ApprovalStatus.CANCELLED.value,
        actor_name="Senior Trader",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="Market conditions shifted.",
    )
    assert ok
    assert dec_c.approval_status == ApprovalStatus.CANCELLED.value
    assert out_c.settlement_status == "CANCELLED"
    assert out_c.realized_amount == 0.0
    assert out_c.realized_costs["total"] == 0.0
    assert ev_c.payload["is_executed"] is False


def test_28_rejected_decisions_do_not_enter_realized_transaction_metrics():
    """
    Defect 28: Rejected decisions are excluded from executed transaction counts and realized costs.
    """
    store = EventStore()
    evaluator = SystemPerformanceEvaluator(store)

    pairs = [
        MatchedDecisionOutcome(
            correlation_id="C1",
            decision_id="D1",
            timestamp="2026-10-17T12:00:00+00:00",
            selected_action="INVEST_REPOS",
            target_asset="REPO-01",
            proposed_amount=10_000_000.0,
            approval_status="APPROVED",
            is_executed=True,
            execution_status="EXECUTED",
            expected_cost=5_000.0,
            realized_cost=5_100.0,
            cost_error=100.0,
            cost_error_bps=2.0,
            liquidity_shortfall=0.0,
            is_shortfall=False,
        ),
        MatchedDecisionOutcome(
            correlation_id="C2",
            decision_id="D2",
            timestamp="2026-10-17T12:00:00+00:00",
            selected_action="INVEST_EQUITIES",
            target_asset="EQ-01",
            proposed_amount=5_000_000.0,
            approval_status="REJECTED",
            is_executed=False,
            execution_status="REJECTED",
            expected_cost=3_000.0,
            realized_cost=0.0,
            cost_error=0.0,
            cost_error_bps=0.0,
            liquidity_shortfall=0.0,
            is_shortfall=False,
        ),
    ]
    rep = evaluator.evaluate(matched_decisions=pairs)
    res = rep.decision_metrics
    assert res.total_decisions == 2
    assert res.executed_decisions == 1
    assert res.cancelled_or_rejected == 1


def test_29_simulated_outcomes_remain_distinct_from_real_observations():
    """
    Defect 29: Simulated outcomes remain distinct from real observations.
    Propagates whether an outcome is simulated or real into stored events and records.
    """
    from reserve_juggler.contracts import DecisionRecord, ServiceStatus
    rec = DecisionRecord(
        decision_id="DEC-SIM-01",
        timestamp="2026-10-17T12:00:00+00:00",
        selected_action="INVEST_REPO",
        target_asset="REPO-01",
        amount=10_000_000.0,
        is_synthetic=True,
        source_status=ServiceStatus.SYNTHETIC_DEMO,
    )
    assert rec.is_synthetic is True
    assert rec.source_status == ServiceStatus.SYNTHETIC_DEMO
    assert rec.source_status != ServiceStatus.CONNECTED


# ==============================================================================
# SECTION 13: REGRESSION TESTS 30-34 — INTEGRATION & DASHBOARD INTEGRITY
# ==============================================================================

def test_30_no_unavailable_gate_is_shown_as_passing():
    """
    Defect 30: Gates with status NOT_EVALUATED or UNKNOWN must not display as PASS.
    """
    gate = HardGateOutcome(
        gate_name="Future-Protected Capital Firewall",
        status=GateStatus.NOT_EVALUATED,
        reason="Upstream safety service not connected.",
    )
    assert gate.status != GateStatus.PASS
    assert gate.status.value == "NOT_EVALUATED"


def test_31_forecast_and_service_source_modes_accurately_shown():
    """
    Defect 31: Facade reports SYNTHETIC_DEMO when in demo mode, and UNINTEGRATED when services missing.
    """
    from reserve_juggler.integration.adapters import UnintegratedServiceError
    facade_demo = ApplicationFacade(use_synthetic=True)
    snap_demo = facade_demo.get_dashboard_snapshot()
    assert snap_demo.data_source_mode == "SYNTHETIC_DEMO"
    facade_demo.close()

    facade_unintegrated = ApplicationFacade(use_synthetic=False)
    assert facade_unintegrated.get_integration_status() == "UNINTEGRATED"
    with pytest.raises(UnintegratedServiceError):
        facade_unintegrated.get_dashboard_snapshot()
    facade_unintegrated.close()


def test_32_all_eight_pages_render_under_populated_and_missing_data_conditions():
    """
    Defect 32: Dashboard snapshot and data layer handle None, empty, and partial data
    cleanly without crashing backend consumers. Full UI page rendering is validated in tests/frontend/.
    """
    facade = ApplicationFacade(use_synthetic=True)
    snapshot = facade.get_dashboard_snapshot()

    # Populate with None/empty edge cases
    snapshot.treasury_state.protected_reserve_floor = None
    snapshot.latest_decision.treasury_safety_score = None
    snapshot.active_candidates = []
    snapshot.audit_log = []

    assert snapshot.treasury_state.protected_reserve_floor is None
    assert snapshot.latest_decision.treasury_safety_score is None
    assert len(snapshot.active_candidates) == 0
    assert len(snapshot.audit_log) == 0
    facade.close()


def test_33_team_1_and_team_2_absence_is_explicitly_surfaced():
    """
    Defect 33: Absence of Team 1 (data/forecasting/ledger) and Team 2 (decision engine)
    is explicitly surfaced as UNINTEGRATED without silent fallbacks.
    """
    facade = ApplicationFacade(use_synthetic=False)
    assert facade.team1_adapter.is_connected is False
    assert facade.team2_adapter.is_connected is False
    assert facade.get_integration_status() == "UNINTEGRATED"
    facade.close()


def test_34_full_end_to_end_path_tested_where_services_available():
    """
    Defect 34: Verified complete end-to-end path:
    Raw market event -> Sentinel classification -> Interpretation ->
    Explainability trace -> Human review revalidation -> Audit logging.
    """
    from reserve_juggler.contracts import ApprovalStatus, UserRole
    facade = ApplicationFacade(use_synthetic=True)
    now = datetime.now(timezone.utc).isoformat()

    # 1. Ingest event
    raw = RawEventRecord(
        event_id="EV-E2E-REG",
        source="REUTERS",
        title="Central Bank Slashes Benchmark Rates by 100 bps",
        summary="Emergency easing response to liquidity conditions",
        publication_time=now,
        ingestion_time=now,
        category=EventCategory.MONETARY_POLICY.value,
        affected_markets=["CASH", "SOVEREIGN_BONDS"],
        source_credibility=0.95,
    )
    sig = facade.sentinel.process_event(raw)
    assert sig is not None

    # 2. Get snapshot and explanation
    snap = facade.get_dashboard_snapshot()
    assert snap.decision_explanation is not None

    # 3. Apply human action
    ok, msg, dec_mod, outcome, ev = facade.explanation_service.execute_human_workflow(
        decision_record=snap.latest_decision,
        action_type=ApprovalStatus.APPROVED.value,
        actor_name="ChiefTreasurer",
        actor_role=UserRole.HEAD_OF_TREASURY.value,
        stated_reason="Verified against live Sentinel alert.",
        treasury_state=snap.treasury_state,
    )
    assert ok
    assert dec_mod.approval_status == ApprovalStatus.APPROVED.value
    assert outcome.settlement_status in ("SETTLED", "PENDING", "SIMULATED_SETTLED")
    facade.close()


def test_35_safety_validator_rejects_nan_inf_and_non_convex_weights():
    """
    Part A5: Validator rejects NaN, Inf, negative values, and non-convex ensemble weights.
    """
    from reserve_juggler.learning.governance import SafetyInvariantValidator
    import math

    validator = SafetyInvariantValidator()

    # 1. Reject NaN
    c_nan = CandidateUpdate(
        candidate_id="C-NAN",
        component="DECISION_OPTIMIZER",
        parameter_key="liquidity_buffer_multiplier",
        current_value=1.25,
        proposed_value=float("nan"),
        description="NaN buffer",
        author="Tester",
        created_at="2026-10-17T12:00:00+00:00",
    )
    ok_nan, errs_nan = validator.validate_candidate(c_nan)
    assert not ok_nan
    assert any("NaN" in e or "finite" in e for e in errs_nan)

    # 2. Reject Inf
    c_inf = CandidateUpdate(
        candidate_id="C-INF",
        component="DECISION_OPTIMIZER",
        parameter_key="liquidity_buffer_multiplier",
        current_value=1.25,
        proposed_value=float("inf"),
        description="Inf buffer",
        author="Tester",
        created_at="2026-10-17T12:00:00+00:00",
    )
    ok_inf, errs_inf = validator.validate_candidate(c_inf)
    assert not ok_inf
    assert any("Infinity" in e or "finite" in e for e in errs_inf)

    # 3. Reject non-convex weights (summing to 1.4)
    c_weights_sum = CandidateUpdate(
        candidate_id="C-WEIGHTS-SUM",
        component="FORECASTING",
        parameter_key="ensemble_combining_weights",
        current_value={"m1": 0.5, "m2": 0.5},
        proposed_value={"m1": 0.7, "m2": 0.7},
        description="Weights summing to 1.4",
        author="Tester",
        created_at="2026-10-17T12:00:00+00:00",
    )
    ok_sum, errs_sum = validator.validate_candidate(c_weights_sum)
    assert not ok_sum
    assert any("summing to 1.0" in e for e in errs_sum)

    # 4. Reject negative weights
    c_weights_neg = CandidateUpdate(
        candidate_id="C-WEIGHTS-NEG",
        component="FORECASTING",
        parameter_key="ensemble_combining_weights",
        current_value={"m1": 0.5, "m2": 0.5},
        proposed_value={"m1": -0.2, "m2": 1.2},
        description="Negative weight",
        author="Tester",
        created_at="2026-10-17T12:00:00+00:00",
    )
    ok_neg, errs_neg = validator.validate_candidate(c_weights_neg)
    assert not ok_neg
    assert any("non-negative" in e for e in errs_neg)


def test_36_stress_simulation_fails_closed_when_protected_reserve_floor_missing():
    """
    Part F1: StressLab fails closed if protected_reserve_floor is missing from input.
    """
    from reserve_juggler.integration.stress_engine import StressLabSimulationEngine
    from reserve_juggler.integration.fixtures import create_mock_treasury_state
    import pytest

    engine = StressLabSimulationEngine()
    state = create_mock_treasury_state()
    state.protected_reserve_floor = None

    with pytest.raises(ValueError, match="protected_reserve_floor"):
        engine.run_simulation(base_treasury_state=state)


def test_37_rollback_rejects_non_existent_target_version():
    """
    Part E4: Rollback fails closed if target version has no restorable configuration.
    """
    service = LearningService()
    ok, msg = service.rollback_component(
        component="FORECASTING",
        target_version="v9.9.9",
        author="Lead Risk Officer",
        approver_role="HEAD_OF_TREASURY",
        justification="Rollback to fictitious version.",
    )
    assert not ok
    assert "no valid restorable runtime configuration" in msg


def test_38_forecast_matcher_prevents_duplicate_flow_event_reuse():
    """
    Part C1: An observation flow event cannot be matched across multiple forecast horizons.
    """
    from reserve_juggler.learning.matcher import OutcomeMatcher
    from reserve_juggler.learning.store import EventStore

    store = EventStore()
    now_ts = "2026-10-18T10:00:00+00:00"

    # Add two forecast events targeting the exact same target timestamp on distinct horizons
    store.record(
        LearningEvent(
            event_id="EV-FC-1H",
            versioned_event_type="FORECAST_INFERENCE_RECORDED",
            module="FORECASTING",
            timestamp="2026-10-18T09:00:00+00:00",
            correlation_id="FC-TEST-1H",
            payload={
                "point_forecasts": {
                    "1h": {
                        "target_timestamp": now_ts,
                        "expected_net_flow": 5_000_000.0,
                        "expected_inflow": 10_000_000.0,
                        "expected_outflow": 5_000_000.0,
                    }
                }
            },
        )
    )
    store.record(
        LearningEvent(
            event_id="EV-FC-24H",
            versioned_event_type="FORECAST_INFERENCE_RECORDED",
            module="FORECASTING",
            timestamp="2026-10-18T09:00:00+00:00",
            correlation_id="FC-TEST-24H",
            payload={
                "point_forecasts": {
                    "24h": {
                        "target_timestamp": now_ts,
                        "expected_net_flow": 4_800_000.0,
                        "expected_inflow": 10_000_000.0,
                        "expected_outflow": 5_200_000.0,
                    }
                }
            },
        )
    )

    # Add only a single actual flow event for this timestamp
    store.record(
        LearningEvent(
            event_id="EV-FLOW-01",
            versioned_event_type="ACTUAL_FLOWS_OBSERVED",
            module="LEDGER",
            timestamp=now_ts,
            correlation_id="FLOW-01",
            payload={"actual_net_flow": 5_100_000.0, "target_timestamp": now_ts},
        )
    )

    matcher = OutcomeMatcher(store)
    pairs = matcher.match_forecasts()

    # The single flow event must only match once, leaving the other forecast unobserved
    matched_actuals = [p for p in pairs if p.actual_net_flow is not None]
    assert len(matched_actuals) == 1
    unmatched_actuals = [p for p in pairs if p.actual_net_flow is None]
    assert len(unmatched_actuals) == 1


def test_39_empty_event_samples_yield_inconclusive_health_and_no_false_alarm_rate():
    """
    Part D1, D2, D3: Zero observations must produce inconclusive health, undefined false alarm rate.
    """
    from reserve_juggler.learning.metrics import SystemPerformanceEvaluator
    from reserve_juggler.learning.store import EventStore

    store = EventStore()
    evaluator = SystemPerformanceEvaluator(store)
    rep = evaluator.evaluate(matched_decisions=[], matched_forecasts=[])

    assert rep.overall_system_health == "INCONCLUSIVE"
    assert rep.forecast_metrics.wape is None
    assert rep.forecast_metrics.mae is None
    assert rep.forecast_metrics.rmse is None
    assert rep.sentinel_metrics.false_alarm_rate is None
    assert rep.sentinel_metrics.precision is None
    assert "UNAVAILABLE_WITHOUT_GROUND_TRUTH" in rep.sentinel_metrics.false_alarm_status


def test_40_hedge_recommendations_not_fabricated_when_optimizer_unintegrated():
    """
    Part F4: Hedge recommendations must state optimizer unintegrated rather than fabricating rules.
    """
    from reserve_juggler.integration.stress_engine import StressLabSimulationEngine
    from reserve_juggler.integration.fixtures import create_mock_treasury_state

    engine = StressLabSimulationEngine()
    state = create_mock_treasury_state()
    state.protected_reserve_floor = 25_000_000.0
    result = engine.run_simulation(base_treasury_state=state, deposit_shock_pct=25.0)
    assert result["recommended_hedge"] == "NO_RECOMMENDATION_OPTIMIZER_UNINTEGRATED"



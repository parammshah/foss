"""
Integration tests for System-Wide Self-Learning Feedback Loop.
Verifies event store persistence, prediction matching, drift detection,
safety invariant enforcement (candidate rejections), and audited rollback.
"""

import pytest
from datetime import datetime, timezone
from reserve_juggler.learning.service import LearningService
from reserve_juggler.learning.models import CandidateStatus
from reserve_juggler.integration.contracts import LearningEvent


def test_event_store_persistence_and_correlation():
    """Verifies feedback event recording and correlation ID retrieval."""
    service = LearningService()
    corr_id = "CORR-TEST-123"
    now = datetime.now(timezone.utc).isoformat()

    ev1 = LearningEvent(
        event_id="EV-1",
        versioned_event_type="FORECAST_INFERENCE_RECORDED",
        module="FORECASTING",
        timestamp=now,
        correlation_id=corr_id,
        payload={"expected_outflow": 10_000_000.0},
    )
    ev2 = LearningEvent(
        event_id="EV-2",
        versioned_event_type="DECISION_GENERATED",
        module="DECISION_OPTIMIZER",
        timestamp=now,
        correlation_id=corr_id,
        payload={"action": "DRAW_REPO", "amount": 10_000_000.0},
    )
    ev3 = LearningEvent(
        event_id="EV-3",
        versioned_event_type="OUTCOME_OBSERVED",
        module="EXECUTION",
        timestamp=now,
        correlation_id=corr_id,
        payload={"realized_amount": 10_000_000.0, "liquidity_shortfall": 0.0},
    )

    service.record(ev1)
    service.record(ev2)
    service.record(ev3)

    correlated = service.store.get_by_correlation_id(corr_id)
    assert len(correlated) == 3
    assert [e.module for e in correlated] == ["FORECASTING", "DECISION_OPTIMIZER", "EXECUTION"]


def test_predictions_matched_to_subsequent_outcomes():
    """Verifies that decisions are accurately paired with observed execution outcomes."""
    service = LearningService()
    corr_id = "CORR-MATCH-456"
    now = datetime.now(timezone.utc).isoformat()

    service.record(
        LearningEvent(
            event_id="EV-DEC",
            versioned_event_type="DECISION_GENERATED",
            module="DECISION_OPTIMIZER",
            timestamp=now,
            correlation_id=corr_id,
            payload={
                "decision_id": corr_id,
                "amount": 15_000_000.0,
                "selected_action": "DRAW_REPO_30D",
                "target_asset": "REPO-30D-GCF",
                "estimated_costs": {"total": 8_000.0},
            },
        )
    )

    service.record(
        LearningEvent(
            event_id="EV-OUT",
            versioned_event_type="OUTCOME_OBSERVED",
            module="EXECUTION",
            timestamp=now,
            correlation_id=corr_id,
            payload={
                "realized_amount": 15_000_000.0,
                "realized_costs": {"total": 8_120.0},
                "liquidity_shortfall": 0.0,
                "settlement_status": "SETTLED",
            },
        )
    )

    pairs = service.matcher.match_all()
    assert len(pairs) == 1
    pair = pairs[0]
    assert pair.correlation_id == corr_id
    assert pair.expected_cost == 8_000.0
    assert pair.realized_cost == 8_120.0
    assert pair.cost_error == 120.0
    assert not pair.is_shortfall


def test_safety_invariants_reject_mandatory_reserve_reduction():
    """
    CRITICAL RULE: The learning system must never silently remove or lower
    mandatory liquidity reserves below statutory floors.
    """
    service = LearningService()

    # Attempt to propose lowering the reserve floor
    candidate = service.propose_update(
        component="FORECASTING",
        parameter_key="mandatory_reserve_floor_pct",
        current_value=0.25,
        proposed_value=0.15,  # Reduction attempt
        description="Optimize idle cash by reducing regulatory reserve floor.",
        author="Aggressive Trading Desk",
        holdout_sample_size=20,
        stress_shortfall_count=0,
    )

    assert candidate.status == CandidateStatus.REJECTED
    assert not candidate.safety_check_passed
    assert any("mandatory liquidity reserve floor" in v for v in candidate.safety_violations)


def test_safety_invariants_reject_candidate_increasing_shortfall():
    """
    CRITICAL RULE: A candidate that improves predictive accuracy but increases
    liquidity shortfalls or funding risk must not be promoted.
    """
    service = LearningService()

    candidate = service.propose_update(
        component="DECISION_OPTIMIZER",
        parameter_key="liquidity_buffer_multiplier",
        current_value=1.5,
        proposed_value=1.1,
        description="Lean buffer to reduce financing drag.",
        author="Optimization Bot",
        holdout_sample_size=30,
        stress_shortfall_count=2,  # Induces 2 shortfalls!
    )

    assert candidate.status == CandidateStatus.REJECTED
    assert not candidate.safety_check_passed
    assert any("liquidity shortfalls" in v for v in candidate.safety_violations)


def test_safety_invariants_reject_insufficient_sample_size():
    """
    CRITICAL RULE: Where insufficient outcome data exist, report that learning is inconclusive
    instead of fabricating improvement.
    """
    service = LearningService()

    candidate = service.propose_update(
        component="FORECASTING",
        parameter_key="arimax_lag_order",
        current_value=3,
        proposed_value=5,
        description="Higher order lag model.",
        author="Research Bot",
        holdout_sample_size=1,  # Insufficient sample!
        stress_shortfall_count=0,
    )

    assert candidate.status == CandidateStatus.REJECTED
    assert any("Insufficient" in v or "insufficient" in v for v in candidate.safety_violations)


def test_candidate_promotion_and_authorized_rollback():
    """Tests full candidate promotion lifecycle and subsequent audited rollback."""
    service = LearningService()

    # Valid candidate
    cand = service.propose_update(
        component="FORECASTING",
        parameter_key="ensemble_combining_weights",
        current_value={"m1": 0.5, "m2": 0.5},
        proposed_value={"m1": 0.4, "m2": 0.6},
        description="Fine-tuned weights on validated holdout.",
        author="Lead Quant",
        holdout_sample_size=50,
        stress_shortfall_count=0,
    )
    assert cand.status == CandidateStatus.SHADOW_MODE

    # Try unauthorized promotion by TREASURY_ANALYST
    ok, err = service.promote_candidate(
        candidate_id=cand.candidate_id,
        approver="Junior Analyst",
        approver_role="TREASURY_ANALYST",
        justification="Looks good.",
    )
    assert not ok
    assert "Permission Denied" in err

    # Authorized promotion by HEAD_OF_TREASURY
    ok_prom, msg_prom = service.promote_candidate(
        candidate_id=cand.candidate_id,
        approver="Chief Treasurer",
        approver_role="HEAD_OF_TREASURY",
        justification="Validated on 50 holdout steps with 0 shortfalls.",
        new_version_tag="v1.5.0",
    )
    assert ok_prom
    assert service.registry.get_active_version("FORECASTING") == "v1.5.0"

    # Authorized Rollback to v1.4.0
    ok_roll, msg_roll = service.rollback_component(
        component="FORECASTING",
        target_version="v1.4.0",
        author="Risk Officer",
        approver_role="RISK_OFFICER",
        justification="Emergency rollback for verification.",
    )
    assert ok_roll
    assert service.registry.get_active_version("FORECASTING") == "v1.4.0"


def test_forecast_matching_is_separate_from_decision_amounts():
    """
    CRITICAL PROMPT REQUIREMENT: Never match a forecast to an unrelated decision amount.
    Match a saved forecast to the actual observed deposits/withdrawals for the target horizon.
    """
    service = LearningService()
    now_dt = datetime.now(timezone.utc)
    target_dt = now_dt.isoformat()
    fc_corr = "FC-HOURLY-001"

    # Forecast event
    service.record(
        LearningEvent(
            event_id="EV-FC-1",
            versioned_event_type="FORECAST_INFERENCE_RECORDED",
            module="FORECASTING",
            timestamp=now_dt.isoformat(),
            correlation_id=fc_corr,
            payload={
                "horizons": ["1h"],
                "quality_status": "HIGH",
                "point_forecasts": {
                    "1h": {
                        "expected_net_flow": -10_000_000.0,
                        "expected_inflow": 5_000_000.0,
                        "expected_outflow": 15_000_000.0,
                        "uncertainty_p05": -18_000_000.0,
                        "uncertainty_p95": -2_000_000.0,
                    }
                },
            },
            associated_versions={"forecasting": "v1.4.0"},
        )
    )

    # Actual cash flow observation
    service.record(
        LearningEvent(
            event_id="EV-FLOW-1",
            versioned_event_type="ACTUAL_CASH_FLOW_OBSERVED",
            module="LEDGER",
            timestamp=target_dt,
            correlation_id=fc_corr,
            payload={
                "actual_net_flow": -11_000_000.0,
                "actual_inflow": 4_500_000.0,
                "actual_outflow": 15_500_000.0,
                "data_quality_status": "VALID",
            },
        )
    )

    matched_fc = service.matcher.match_forecasts()
    assert len(matched_fc) == 1
    m = matched_fc[0]
    assert m.predicted_outflow == 15_000_000.0
    assert m.actual_outflow == 15_500_000.0
    assert m.abs_error == 1_000_000.0
    # Within 90% uncertainty interval [-18M, -2M]
    assert m.within_90_interval is True


def test_metrics_zero_actuals_mape_and_wape_reported():
    """Verifies that zero/near-zero actual outflows do not crash or produce infinite MAPE, and WAPE is reported."""
    from reserve_juggler.learning.metrics import calculate_forecast_metrics
    from reserve_juggler.learning.models import MatchedForecastOutcome

    # Zero actual outflow
    m_zero = MatchedForecastOutcome(
        correlation_id="C-ZERO",
        forecast_id="FC-Z",
        target_timestamp="2026-10-17T12:00:00Z",
        horizon="1h",
        predicted_net_flow=0.0,
        predicted_inflow=0.0,
        predicted_outflow=500_000.0,
        uncertainty_p05=-1_000_000.0,
        uncertainty_p50=0.0,
        uncertainty_p90=500_000.0,
        uncertainty_p95=1_000_000.0,
        uncertainty_p99=1_500_000.0,
        model_version="v1.0",
        actual_net_flow=0.0,
        actual_inflow=0.0,
        actual_outflow=0.0,  # Zero actual!
        is_eligible=True,
    )

    metrics = calculate_forecast_metrics([m_zero])
    assert metrics.valid_evaluations == 1
    assert metrics.excluded_count == 1
    assert any("Zero actual" in r for r in metrics.exclusion_reasons)
    assert metrics.mape is None  # Handled safely as None instead of inf/nan
    assert metrics.mae == 500_000.0
    assert metrics.wape == 100.0  # Safe WAPE


def test_drift_detection_statistical_ks_test():
    """Verifies Kolmogorov-Smirnov two-sample drift detection on forecast errors."""
    from reserve_juggler.learning.metrics import detect_distribution_drift

    # Stable errors from identical distribution
    ref_errors = [10.0, 12.0, 11.5, 9.8, 10.2, 11.0, 10.5, 12.1, 9.9, 10.8]
    test_errors_stable = [10.2, 11.8, 11.0, 10.0, 10.4, 11.2, 10.7, 11.9, 10.1, 10.6]

    res_stable = detect_distribution_drift(ref_errors, test_errors_stable)
    assert res_stable.status == "STABLE"
    assert not res_stable.drift_detected
    assert res_stable.p_value > 0.05

    # Shifted distribution (severe drift)
    test_errors_drift = [55.0, 62.0, 58.0, 70.0, 64.0, 59.0, 66.0, 72.0, 61.0, 68.0]
    res_drift = detect_distribution_drift(ref_errors, test_errors_drift)
    assert res_drift.status == "DRIFT_CONFIRMED"
    assert res_drift.drift_detected
    assert res_drift.p_value < 0.05


def test_promotion_updates_operational_runtime_parameters():
    """
    CRITICAL PROMPT REQUIREMENT: Updating a version number alone is not a deployment.
    Promoting a candidate must update the actual registered operational runtime parameters.
    """
    service = LearningService()
    initial_rt_val = service.runtime_registry.get_runtime_parameter("FORECASTING", "ensemble_combining_weights")

    cand = service.propose_update(
        component="FORECASTING",
        parameter_key="ensemble_combining_weights",
        current_value=initial_rt_val,
        proposed_value={"lstm_weight": 0.45, "arimax_weight": 0.55},
        description="Updated convex weights.",
        author="Lead Quant",
        holdout_sample_size=30,
        stress_shortfall_count=0,
    )

    ok, msg = service.promote_candidate(
        candidate_id=cand.candidate_id,
        approver="Chief Treasurer",
        approver_role="HEAD_OF_TREASURY",
        justification="Verified holdout performance.",
        new_version_tag="v2.0.0",
    )
    assert ok
    # Runtime state must now be the proposed value!
    active_param = service.runtime_registry.get_runtime_parameter("FORECASTING", "ensemble_combining_weights")
    assert active_param == {"lstm_weight": 0.45, "arimax_weight": 0.55}

    # Rollback must restore initial state
    ok_rb, _ = service.rollback_component(
        component="FORECASTING",
        target_version="v1.4.0",
        author="Risk Officer",
        approver_role="RISK_OFFICER",
        justification="Rollback test.",
    )
    assert ok_rb
    restored_param = service.runtime_registry.get_runtime_parameter("FORECASTING", "ensemble_combining_weights")
    assert restored_param == initial_rt_val


def test_durable_sqlite_persistence_survives_reinstantiation(tmp_path):
    """Verifies that SQLite event store persists records across database reopening."""
    from reserve_juggler.learning.sqlite_store import SqliteEventStore
    from reserve_juggler.contracts import LearningEvent

    db_file = str(tmp_path / "test_events.db")
    store1 = SqliteEventStore(db_path=db_file)

    ev = LearningEvent(
        event_id="EV-DURABLE-001",
        versioned_event_type="DECISION_GENERATED",
        module="DECISION_OPTIMIZER",
        timestamp="2026-10-17T12:00:00Z",
        correlation_id="DEC-DURABLE-001",
        payload={"action": "DRAW_REPO", "amount": 10_000_000.0},
    )
    store1.record_event(ev)

    # Re-instantiate a new store pointing to the same file
    store2 = SqliteEventStore(db_path=db_file)
    events = store2.get_all_events()
    assert len(events) == 1
    assert events[0].event_id == "EV-DURABLE-001"
    assert events[0].payload["amount"] == 10_000_000.0


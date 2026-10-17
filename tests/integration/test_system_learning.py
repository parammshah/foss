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

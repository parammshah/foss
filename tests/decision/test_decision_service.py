"""
Tests for TreasuryDecisionService orchestration and event publishing.
"""

import json
from datetime import datetime, timedelta, timezone

import pytest

from reserve_juggler.decision.events import InMemoryEventPublisher
from reserve_juggler.decision.service import TreasuryDecisionService
from reserve_juggler.tax.tax_profile import create_default_corporate_profile
from tests.decision.fixtures import (
    ConcreteExecutionOutcome,
    ConcreteObligation,
    ConcreteRiskSignal,
    ConcreteTreasuryState,
    make_benchmark_repo_bridge_state,
    make_healthy_forecast_bundle,
)


def test_decision_service_decide_and_json_serializability():
    """Verify TreasuryDecisionService.decide generates valid JSON-serializable DecisionRecord."""
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)
    state = make_benchmark_repo_bridge_state(now, credit_available=15000000.0)
    tax_profile = create_default_corporate_profile()

    publisher = InMemoryEventPublisher()
    service = TreasuryDecisionService(event_publisher=publisher)

    record = service.decide(
        state=state,
        forecast_bundle=bundle,
        risk_signals=[],
        tax_profile=tax_profile,
    )

    # 1. Structural assertions on DecisionRecord
    assert record.decision_id.startswith("dec-")
    assert record.selected_action["action_type"] == "DRAW_BRIDGE_CREDIT"
    assert len(record.alternatives) > 0
    assert "composite_score" in record.score
    assert "protected_capital" in record.reserve
    assert record.eligible_surplus >= 0.0
    assert len(record.reason_codes) > 0

    # 2. Strict JSON serialization test
    record_dict = record.to_dict()
    json_str = json.dumps(record_dict)
    assert len(json_str) > 0

    # 3. Verify Team 3 learning events were published
    assert len(publisher.events) > 0
    event_types = [e.versioned_event_type for e in publisher.events]
    assert "DECISION_PROPOSED_V1" in event_types


def test_decision_service_publishes_rejection_events():
    """Verify rejected actions (e.g. payout violations or credit overflows) generate ACTION_REJECTED events."""
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)
    state = make_benchmark_repo_bridge_state(now, credit_available=0.0)

    publisher = InMemoryEventPublisher()
    service = TreasuryDecisionService(event_publisher=publisher)

    service.decide(state=state, forecast_bundle=bundle)

    # Check for ACTION_REJECTED_V1 events
    rejected_events = [
        e for e in publisher.events if e.versioned_event_type == "ACTION_REJECTED_V1"
    ]
    assert len(rejected_events) >= 1
    # Verify rejection reason mentions delay or credit limit
    assert any("REJECTED" in e.structured_data["reason"] for e in rejected_events)


def test_decision_service_record_execution_outcome():
    """Verify execution outcome feedback publishes event for Team 3."""
    publisher = InMemoryEventPublisher()
    service = TreasuryDecisionService(event_publisher=publisher)

    outcome = ConcreteExecutionOutcome(
        realized_cash_movements=10000000.0,
        costs=273.97,
        penalties=0.0,
        losses=0.0,
        liquidity_shortfalls=0.0,
        metadata={"settlement_confirmed": True},
    )

    event = service.record_execution_outcome(decision_id="dec-123456", outcome=outcome)

    assert event.versioned_event_type == "EXECUTION_OUTCOME_RECORDED_V1"
    assert event.correlation_id == "dec-123456"
    assert event.structured_data["costs"] == 273.97
    assert event.structured_data["liquidity_shortfalls"] == 0.0
    assert event in publisher.events

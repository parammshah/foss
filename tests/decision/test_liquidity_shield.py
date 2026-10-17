"""
Tests for Adaptive Liquidity Shield and Hard Safety Gates.
"""

from datetime import datetime, timedelta, timezone

import pytest

from reserve_juggler.risk.liquidity_shield import (
    assess_liquidity_reserve,
    evaluate_hard_safety_gates,
)
from tests.decision.fixtures import (
    ConcreteAssetPosition,
    ConcreteObligation,
    ConcreteTreasuryState,
    make_healthy_forecast_bundle,
)


def test_liquidity_reserve_assessment():
    """Verify dynamic reserve calculation without double counting."""
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    # State: $15M cash, $2M immediate obligation (due in 6h), $5M maturing repo in 12h
    ob = ConcreteObligation(
        due_time=now + timedelta(hours=6),
        amount=2000000.0,
        is_mandatory=True,
    )
    repo = ConcreteAssetPosition(
        asset_id="REPO_5M",
        tier=1,
        market_value=5000000.0,
        liquidatable_value=5000000.0,
        availability=now + timedelta(hours=12),
        maturity=now + timedelta(hours=12),
        annual_yield=0.048,
        instrument_type="repo",
    )

    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=15000000.0,
        asset_positions=[repo],
        future_obligations=[ob],
    )

    assessment = assess_liquidity_reserve(
        state, bundle, operational_safety_buffer=0.10, payout_window_hours=24.0
    )

    # Immediate obligation $2M + 10% buffer ($200k) = $2.2M + 0-24h forecast deficit
    assert assessment.immediate_obligations == 2000000.0
    assert assessment.operational_safety_buffer == 200000.0
    assert assessment.protected_capital >= 2200000.0
    # Eligible surplus = Cash - Protected Capital - Future Obligation Reserve
    assert assessment.eligible_investable_surplus <= 15000000.0 - assessment.protected_capital
    assert assessment.eligible_investable_surplus > 0.0


def test_hard_gate_immediate_payout_coverage_failure():
    """Verify hard gate fails when funding cannot cover immediate payouts."""
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    ob = ConcreteObligation(
        due_time=now + timedelta(hours=2),
        amount=5000000.0,
        is_mandatory=True,
    )

    # State with only $1M cash and $1M credit = $2M funding vs $5M obligations
    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=1000000.0,
        credit_availability=1000000.0,
        future_obligations=[ob],
    )

    reserve = assess_liquidity_reserve(state, bundle)
    gate_eval = evaluate_hard_safety_gates(state, reserve)

    assert gate_eval.passed is False
    assert "immediate_payout_coverage" in gate_eval.violated_gates
    assert any("BREACH_IMMEDIATE_PAYOUT_COVERAGE" in r for r in gate_eval.reason_codes)


def test_hard_gate_rejects_investment_exceeding_surplus():
    """Verify hard gate rejects investment that violates protected reserve."""
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    ob = ConcreteObligation(
        due_time=now + timedelta(hours=4),
        amount=5000000.0,
        is_mandatory=True,
    )

    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=6000000.0,  # $6M cash vs ~$5.5M required reserve
        future_obligations=[ob],
    )

    reserve = assess_liquidity_reserve(state, bundle)
    eligible = reserve.eligible_investable_surplus  # around $500k

    # Propose investing $2M (which exceeds eligible surplus)
    gate_eval = evaluate_hard_safety_gates(
        state,
        reserve,
        proposed_investment_amount=2000000.0,
    )

    assert gate_eval.passed is False
    assert "protected_reserve_intact" in gate_eval.violated_gates
    assert any("BREACH_PROTECTED_RESERVE" in r for r in gate_eval.reason_codes)


def test_hard_gate_rejects_repo_lockup_clashing_with_payout():
    """Verify repo lockup timing gate rejects locking funds needed for intermediate obligations."""
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    # Obligation of $3M due in 4 hours
    ob = ConcreteObligation(
        due_time=now + timedelta(hours=4),
        amount=3000000.0,
        is_mandatory=True,
    )

    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=4000000.0,
        future_obligations=[ob],
    )

    reserve = assess_liquidity_reserve(state, bundle)

    # Proposed 24-hour repo lockup of $2.5M
    # Remaining free cash = $4M - $2.5M = $1.5M, but $3M obligation is due in 4 hours!
    gate_eval = evaluate_hard_safety_gates(
        state,
        reserve,
        proposed_investment_amount=2500000.0,
        proposed_lockup_hours=24.0,
    )

    assert gate_eval.passed is False
    assert "lockup_timing_feasibility" in gate_eval.violated_gates


def test_hard_gate_credit_limit_respected():
    """Verify proposed bridge draw cannot exceed authorized limit."""
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=0.0,
        credit_availability=1000000.0,  # $1M available
    )
    reserve = assess_liquidity_reserve(state, bundle)

    # Request $2M draw
    gate_eval = evaluate_hard_safety_gates(
        state,
        reserve,
        proposed_bridge_draw=2000000.0,
    )

    assert gate_eval.passed is False
    assert "credit_availability_limit" in gate_eval.violated_gates

"""
Tests for Treasury Safety Score (TSS) calculation engine.
"""

from datetime import datetime, timedelta, timezone

import pytest

from reserve_juggler.risk.models import SafetyMode, TSSWeights
from reserve_juggler.risk.safety_score import calculate_treasury_safety_score
from tests.decision.fixtures import (
    ConcreteAssetPosition,
    ConcreteForecastBundle,
    ConcreteObligation,
    ConcreteRiskSignal,
    ConcreteTreasuryState,
    make_healthy_forecast_bundle,
)


def test_tss_healthy_optimizing_treasury():
    """Verify healthy treasury achieves Optimizing mode (>=80)."""
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    # Well-capitalized state: $10M cash, $200k obligations, healthy repos
    repo = ConcreteAssetPosition(
        asset_id="REPO_1",
        tier=1,
        market_value=5000000.0,
        liquidatable_value=5000000.0,
        availability=now + timedelta(hours=2),
        maturity=now + timedelta(hours=2),
        settlement_delay_hours=0.0,
        annual_yield=0.048,
        exit_fees=0.001,
        haircut=0.0,
        duration=0.0,
        volatility=0.0,
        cost_basis=5000000.0,
        instrument_type="repo",
    )

    obligation = ConcreteObligation(
        due_time=now + timedelta(hours=12),
        amount=200000.0,
        is_mandatory=True,
    )

    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=10000000.0,
        asset_positions=[repo],
        future_obligations=[obligation],
        credit_availability=5000000.0,
    )

    result = calculate_treasury_safety_score(state, bundle)

    assert result.composite_score >= 80.0
    assert result.mode == SafetyMode.OPTIMIZING
    assert not result.shortfall_detected
    assert not result.is_calibrated_probability
    assert 0.0 <= result.components.liquidity_coverage <= 100.0
    assert 0.0 <= result.components.stress_survival <= 100.0
    assert 0.0 <= result.components.forecast_reliability <= 100.0


def test_tss_circuit_breaker_cash_shortfall():
    """Verify that acute cash shortfall triggers circuit breaker:

    TSS must NEVER conceal a critical cash shortfall behind a high average.
    Score must be capped at Critical (<= 35.0).
    """
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    # State with $10M immediate obligation, but only $100k cash and $0 credit
    # (even if holding $50M in illiquid bonds)
    bond = ConcreteAssetPosition(
        asset_id="BOND_ILLIQUID",
        tier=2,
        market_value=50000000.0,
        liquidatable_value=48000000.0,
        availability=now + timedelta(days=2),
        maturity=now + timedelta(days=730),
        settlement_delay_hours=48.0,
        annual_yield=0.05,
        exit_fees=0.01,
        haircut=0.03,
        duration=2.0,
        volatility=0.02,
        cost_basis=50000000.0,
        instrument_type="bond",
    )

    obligation = ConcreteObligation(
        due_time=now + timedelta(hours=2),
        amount=10000000.0,  # $10M due in 2 hours
        is_mandatory=True,
    )

    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=100000.0,  # Only $100k available!
        asset_positions=[bond],
        future_obligations=[obligation],
        credit_availability=0.0,  # No credit line
    )

    result = calculate_treasury_safety_score(state, bundle)

    # Circuit breaker must fire!
    assert result.shortfall_detected is True
    assert result.composite_score <= 35.0
    assert result.mode == SafetyMode.CRITICAL
    assert any("CRITICAL: Available cash is below immediate payout obligations" in w for w in result.warnings)


def test_tss_stale_forecast_triggers_conservative_ceiling():
    """Verify stale forecast bundle penalizes reliability and caps TSS."""
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = ConcreteForecastBundle(
        point_forecasts=[],
        quality_status="STALE",
    )

    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=5000000.0,
        future_obligations=[],
    )

    result = calculate_treasury_safety_score(state, bundle)

    assert result.components.forecast_reliability <= 35.0
    assert result.composite_score <= 55.0  # Capped due to stale forecast
    assert any("STALE" in w for w in result.warnings)


def test_tss_risk_signal_penalties():
    """Verify high and critical risk signals deduct from reliability and composite score."""
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=5000000.0,
        future_obligations=[],
    )

    signals = [
        ConcreteRiskSignal(
            source="MARKET_VOLATILITY_MONITOR",
            severity="CRITICAL",
            confidence=0.9,
            supporting_evidence={"vix": 42.5},
            scenario_tags=["LIQUIDITY_RUN"],
        ),
        ConcreteRiskSignal(
            source="INTERBANK_LENDING_ALERT",
            severity="HIGH",
            confidence=0.8,
            supporting_evidence={"spread_bps": 120},
            scenario_tags=["CREDIT_FREEZE"],
        ),
    ]

    result = calculate_treasury_safety_score(state, bundle, risk_signals=signals)
    result_clean = calculate_treasury_safety_score(state, bundle, risk_signals=[])

    assert result.components.forecast_reliability < result_clean.components.forecast_reliability
    assert result.composite_score < result_clean.composite_score


def test_tss_weights_validation():
    """Verify invalid weights that do not sum to 1.0 raise ValueError."""
    with pytest.raises(ValueError, match="must sum to 1.0"):
        TSSWeights(liquidity_coverage=0.5, stress_survival=0.8)

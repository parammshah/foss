"""
Tests for Capital Firewall and Holding-Time Gatekeeper.
"""

from datetime import datetime, timezone

import pytest

from reserve_juggler.risk.capital_firewall import (
    calculate_holding_time_gate,
    classify_hot_core_capital,
)
from reserve_juggler.risk.liquidity_shield import assess_liquidity_reserve
from tests.decision.fixtures import (
    ConcreteAssetPosition,
    ConcreteObligation,
    ConcreteTreasuryState,
    make_healthy_forecast_bundle,
)


def test_classify_hot_core_capital():
    """Verify correct economic partitioning between Hot Money and Core Capital."""
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    ob = ConcreteObligation(due_time=now, amount=1000000.0)
    bond = ConcreteAssetPosition(
        asset_id="BOND_2Y",
        tier=2,
        market_value=10000000.0,
        liquidatable_value=9800000.0,
        availability=now,
    )

    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=5000000.0,
        asset_positions=[bond],
        future_obligations=[ob],
    )

    reserve = assess_liquidity_reserve(state, bundle)
    classification = classify_hot_core_capital(state, bundle, reserve)

    assert classification.total_treasury_assets == 15000000.0
    assert classification.hot_money == reserve.protected_capital + reserve.future_obligation_reserve
    assert classification.core_capital == max(0.0, 15000000.0 - classification.hot_money)
    assert classification.investable_core_capital <= reserve.eligible_investable_surplus


def test_bond_holding_time_gatekeeper_benchmark():
    """Verify the Section 6 benchmark formula:

    minimum_days = (0.02 / 0.048) * 365 = 152.0833... days.
    """
    bond = ConcreteAssetPosition(
        asset_id="BENCHMARK_2Y_SOVEREIGN",
        tier=2,
        market_value=10000000.0,
        liquidatable_value=9800000.0,
        availability=datetime(2026, 10, 17, tzinfo=timezone.utc),
        annual_yield=0.048,  # 4.8%
        haircut=0.02,  # 2.0%
        instrument_type="bond",
    )

    # 1. Planned holding of 180 days (greater than ~152.08 days) -> Passes
    result_pass = calculate_holding_time_gate(bond, expected_holding_days=180.0)
    assert result_pass.passed is True
    assert round(result_pass.minimum_holding_days, 1) == 152.1

    # 2. Planned holding of 90 days (less than ~152.08 days) -> Fails
    result_fail = calculate_holding_time_gate(bond, expected_holding_days=90.0)
    assert result_fail.passed is False
    assert result_fail.expected_holding_days == 90.0
    assert "Failed hurdle" in result_fail.notes


def test_equity_holding_time_simplistic_formula_rejected():
    """Verify simplistic formula is strictly rejected for equities as required by Section 6."""
    equity = ConcreteAssetPosition(
        asset_id="TECH_EQUITY",
        tier=3,
        market_value=2000000.0,
        liquidatable_value=1900000.0,
        availability=datetime(2026, 10, 17, tzinfo=timezone.utc),
        annual_yield=0.08,
        haircut=0.05,
        instrument_type="equity",
    )

    result = calculate_holding_time_gate(equity, expected_holding_days=365.0)

    assert result.passed is False
    assert result.minimum_holding_days == float("inf")
    assert "cannot be applied to equities" in result.notes

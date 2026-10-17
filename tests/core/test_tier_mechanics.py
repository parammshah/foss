"""Unit tests for the four asset tiers, maturity mechanics, early exit penalties, and liquidation haircuts."""

from datetime import datetime, timezone, timedelta
import pytest
from reserve_juggler.contracts import AssetPosition, AssetTier
from reserve_juggler.ledger import (
    InitialTreasuryConfig,
    TreasuryLedger,
    calculate_liquidatable_value,
)


def test_liquidatable_value_calculations():
    # Tier 0 Cash: $1M market = $1M liquidatable
    assert calculate_liquidatable_value(AssetTier.TIER_0_CASH, 1_000_000.0, 0.0, 0.0) == 1_000_000.0

    # Tier 1 Repo: $1M market, 0.5% early break penalty
    # If not matured: 1M - 0.5% = 995,000
    assert calculate_liquidatable_value(AssetTier.TIER_1_REPO, 1_000_000.0, 0.005, 0.0, is_matured=False) == 995_000.0
    # If matured: 100% realizable
    assert calculate_liquidatable_value(AssetTier.TIER_1_REPO, 1_000_000.0, 0.005, 0.0, is_matured=True) == 1_000_000.0

    # Tier 2 Bonds: $1M market, 2.0% emergency haircut = 980,000
    assert calculate_liquidatable_value(AssetTier.TIER_2_BONDS, 1_000_000.0, 0.0, 0.02) == 980_000.0

    # Tier 3 Equities: $1M market, 15.0% emergency slippage = 850,000
    assert calculate_liquidatable_value(AssetTier.TIER_3_EQUITIES, 1_000_000.0, 0.0, 0.15) == 850_000.0


def test_repo_sweep_and_auto_maturity_at_0800():
    config = InitialTreasuryConfig(opening_cash_usd=20_000_000.0)
    ledger = TreasuryLedger(config)
    t_1700 = datetime(2026, 1, 1, 17, 0, 0, tzinfo=timezone.utc)
    state_0 = ledger.get_state(t_1700)

    # 1. Sweep $10M into overnight repo at 17:00
    action = {"type": "SWEEP_REPO", "amount_usd": 10_000_000.0}
    state_1, outcome_1 = ledger.simulate_action(state_0, action)

    assert outcome_1.execution_status == "SUCCESS"
    assert state_1.operational_cash == 10_000_000.0
    assert state_1.total_investments_market_value_usd == 10_000_000.0
    # Early break penalty would be 0.5% ($50,000), so liquidatable is $9,950,000
    repo_pos = next(v for v in state_1.asset_positions.values() if v.tier == AssetTier.TIER_1_REPO)
    assert repo_pos.liquidatable_value_usd == pytest.approx(9_950_000.0)

    # Apply to live ledger
    ledger.cash_usd = state_1.operational_cash_usd
    ledger.positions = state_1.asset_positions

    # 2. Advance 15 hours until 08:00 next morning (repo maturity)
    t_0800 = datetime(2026, 1, 2, 8, 0, 0, tzinfo=timezone.utc)
    # Accrue interest for 14 hours
    for h in range(14):
        step_t = t_1700 + timedelta(hours=h+1)
        ledger.process_hourly_step(step_t, 0.0, 0.0, repo_sweep_mature=False)

    # At 08:00: repo_sweep_mature is True!
    net, shortfall = ledger.process_hourly_step(t_0800, 0.0, 0.0, repo_sweep_mature=True)

    # Repo position should be cleared and principal + interest credited back to cash!
    assert len([p for p in ledger.positions.values() if p.tier == AssetTier.TIER_1_REPO]) == 0
    # Interest on $10M at 4.3% APY for 15 hours: ~10M * 0.043 * 15 / 8760 = $736.30
    expected_interest = 10_000_000.0 * 0.043 * (15.0 / 8760.0)
    assert ledger.cash_usd == pytest.approx(20_000_000.0 + expected_interest, rel=1e-3)


def test_early_break_repo_penalty():
    config = InitialTreasuryConfig(opening_cash_usd=10_000_000.0)
    ledger = TreasuryLedger(config)
    now = datetime(2026, 1, 1, 18, 0, 0, tzinfo=timezone.utc)

    # Enter $6M repo
    state_0 = ledger.get_state(now)
    state_1, _ = ledger.simulate_action(state_0, {"type": "SWEEP_REPO", "amount_usd": 6_000_000.0})

    # Break early at 20:00 (before maturity)
    state_2, outcome_break = ledger.simulate_action(state_1, {"type": "BREAK_REPO", "amount_usd": 6_000_000.0})

    assert outcome_break.execution_status == "SUCCESS"
    # Early exit fee is 0.5% of $6M = $30,000
    assert outcome_break.penalties_usd == pytest.approx(30_000.0)
    # Net proceeds = $6M - $30k = $5,970,000 credited to cash
    # Total cash = $4M + $5.97M = $9,970,000
    assert state_2.operational_cash == pytest.approx(9_970_000.0)


def test_bond_and_equity_simulation_actions():
    config = InitialTreasuryConfig(opening_cash_usd=50_000_000.0)
    ledger = TreasuryLedger(config)
    now = datetime(2026, 1, 1, 14, 0, 0, tzinfo=timezone.utc)
    state_0 = ledger.get_state(now)

    # Buy $10M of bonds
    state_bonds, outcome_bonds = ledger.simulate_action(state_0, {"type": "BUY_BONDS", "amount_usd": 10_000_000.0})
    assert outcome_bonds.execution_status == "SUCCESS"
    assert "T2_BONDS_PORTFOLIO" in state_bonds.asset_positions
    pos = state_bonds.asset_positions["T2_BONDS_PORTFOLIO"]
    # 2% liquidation haircut
    assert pos.liquidatable_value_usd == pytest.approx(10_000_000.0 * 0.98)

    # Liquidate bonds (emergency sell)
    state_sold, outcome_sell = ledger.simulate_action(state_bonds, {"type": "SELL_BONDS", "amount_usd": 10_000_000.0})
    assert outcome_sell.execution_status == "SUCCESS"
    assert outcome_sell.losses_usd == pytest.approx(200_000.0)  # 2% haircut realized as loss

    # Buy $5M equities
    state_eq, outcome_eq = ledger.simulate_action(state_0, {"type": "BUY_EQUITIES", "amount_usd": 5_000_000.0})
    assert outcome_eq.execution_status == "SUCCESS"
    eq_pos = state_eq.asset_positions["T3_EQUITIES_INDEX"]
    # 15% slippage
    assert eq_pos.liquidatable_value_usd == pytest.approx(5_000_000.0 * 0.85)

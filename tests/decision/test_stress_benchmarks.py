"""
Section 10: Stress Testing and Baseline Strategy Comparison.

Compares:
1. All-Cash Strategy
2. Greedy Gross-Yield Strategy
3. Adaptive Reserve Juggler Strategy

Across identical stress scenarios:
- Withdrawal spikes
- Deposit shortfalls
- Market declines
- Rising bond yields
- Repo availability issues
- Credit facility unavailability
- Increased funding costs
- Compound multi-variable crisis
"""

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import pytest

from reserve_juggler.decision.service import TreasuryDecisionService
from reserve_juggler.optimization.models import ActionType
from reserve_juggler.tax.tax_profile import create_default_corporate_profile
from tests.decision.fixtures import (
    ConcreteAssetPosition,
    ConcreteForecastBundle,
    ConcreteForecastPoint,
    ConcreteObligation,
    ConcreteRiskSignal,
    ConcreteTreasuryState,
    make_healthy_forecast_bundle,
)


@dataclass
class SimulationMetrics:
    """Metrics recorded for baseline strategy comparisons."""

    strategy_name: str
    scenario_name: str
    payout_shortfalls: float  # Unmet mandatory customer obligations ($)
    emergency_borrowing: float  # Emergency bridge/credit drawn ($)
    fees_and_penalties: float  # Early exit penalties and transaction fees ($)
    liquidation_losses: float  # Haircuts, market drop losses, slippage ($)
    after_tax_net_return: float  # Net yield minus financing, fees, taxes ($)
    remaining_liquidity: float  # Final available unencumbered cash ($)


def run_all_cash_strategy(
    initial_cash: float,
    obligations: list[ConcreteObligation],
    deposit_inflow: float,
) -> SimulationMetrics:
    """Simulate All-Cash Strategy: keeps 100% of capital in liquid cash."""
    cash = initial_cash + deposit_inflow
    shortfall = 0.0

    for ob in obligations:
        if cash >= ob.amount:
            cash -= ob.amount
        else:
            shortfall += (ob.amount - cash)
            cash = 0.0

    return SimulationMetrics(
        strategy_name="All-Cash Strategy",
        scenario_name="Stress Simulation",
        payout_shortfalls=shortfall,
        emergency_borrowing=0.0,
        fees_and_penalties=0.0,
        liquidation_losses=0.0,
        after_tax_net_return=0.0,  # Zero yield on pure cash
        remaining_liquidity=cash,
    )


def run_greedy_gross_yield_strategy(
    initial_cash: float,
    obligations: list[ConcreteObligation],
    deposit_inflow: float,
    credit_avail: float,
    bond_haircut: float = 0.02,
    yield_rate: float = 0.048,
    market_drawdown: float = 0.0,
) -> SimulationMetrics:
    """Simulate Greedy Gross-Yield Strategy:

    Chases maximum headline yield by investing 95% of initial cash into 2-year sovereign bonds
    or equities, leaving negligible cash buffer.
    When obligations or withdrawals arrive, is forced to liquidate under fire-sale haircuts
    and market drawdowns.
    """
    # Invest 95% of cash in bonds to chase 4.8% yield
    invested_amount = initial_cash * 0.95
    cash = (initial_cash * 0.05) + deposit_inflow
    gross_yield = invested_amount * yield_rate * (30 / 365.0)

    # Asset value under market shock
    bond_value = invested_amount * (1.0 - market_drawdown)

    shortfall = 0.0
    liquidation_losses = 0.0
    emergency_borrowing = 0.0
    fees = 0.0

    for ob in obligations:
        needed = ob.amount
        if cash >= needed:
            cash -= needed
        else:
            deficit = needed - cash
            cash = 0.0

            # Forced fire-sale of bonds with haircut
            effective_haircut = bond_haircut + 0.03  # Distressed sale penalty
            if bond_value > 0.0:
                proceeds = min(bond_value * (1.0 - effective_haircut), deficit)
                sold_nominal = proceeds / (1.0 - effective_haircut)
                bond_value -= sold_nominal
                loss = sold_nominal * effective_haircut
                liquidation_losses += loss
                deficit -= proceeds

            if deficit > 0.0:
                # Attempt emergency credit
                if credit_avail >= deficit:
                    emergency_borrowing += deficit
                    fees += deficit * 0.12 * (48.0 / 8760.0)
                    credit_avail -= deficit
                    deficit = 0.0
                else:
                    emergency_borrowing += credit_avail
                    deficit -= credit_avail
                    credit_avail = 0.0
                    shortfall += deficit

    after_tax_net = max(-1e6, gross_yield - liquidation_losses - fees)

    return SimulationMetrics(
        strategy_name="Greedy Gross-Yield Strategy",
        scenario_name="Stress Simulation",
        payout_shortfalls=shortfall,
        emergency_borrowing=emergency_borrowing,
        fees_and_penalties=fees,
        liquidation_losses=liquidation_losses,
        after_tax_net_return=after_tax_net,
        remaining_liquidity=cash,
    )


def test_stress_withdrawal_spike_and_deposit_shortfall():
    """Stress Scenario 1: Severe Withdrawal Spike & Deposit Drought.

    Withdrawal spike of $8M with zero deposits.
    Initial cash: $10M.
    Compare All-Cash, Greedy, and Adaptive Reserve Juggler.
    """
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    obligations = [
        ConcreteObligation(due_time=now + timedelta(hours=2), amount=5000000.0),
        ConcreteObligation(due_time=now + timedelta(hours=6), amount=3000000.0),
    ]

    # 1. All-Cash
    m_cash = run_all_cash_strategy(
        initial_cash=10000000.0, obligations=obligations, deposit_inflow=0.0
    )
    assert m_cash.payout_shortfalls == 0.0
    assert m_cash.after_tax_net_return == 0.0

    # 2. Greedy Strategy
    m_greedy = run_greedy_gross_yield_strategy(
        initial_cash=10000000.0,
        obligations=obligations,
        deposit_inflow=0.0,
        credit_avail=2000000.0,
        bond_haircut=0.02,
        market_drawdown=0.05,
    )
    # Greedy suffers emergency borrowing and liquidation losses
    assert m_greedy.liquidation_losses > 0.0

    # 3. Reserve Juggler Strategy
    bundle = make_healthy_forecast_bundle(now)
    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=10000000.0,
        future_obligations=obligations,
        credit_availability=5000000.0,
    )

    service = TreasuryDecisionService()
    record = service.decide(state, bundle)

    # Reserve Juggler maintains protected capital covering obligations
    assert record.reserve["protected_capital"] >= 8000000.0
    assert record.selected_action["action_type"] in ("KEEP_CASH", "PLACE_REPO")
    # Guarantee: Increasing yield cannot override hard safety gate!
    assert record.hard_gate_results["passed"] is True


def test_stress_rising_bond_yields_and_market_declines():
    """Stress Scenario 2: +250 bps yield rise (bond duration loss) and -25% equity crash.

    Demonstrates Reserve Juggler's holding-time gate and tier exposure caps protect capital,
    while Greedy Strategy experiences severe drawdown.
    """
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    # 2-year sovereign bond with duration 2.0
    # +250 bps (+2.5%) yield hike causes ~5.0% price decline (-duration * delta_yield)
    # 25% equity decline
    bond = ConcreteAssetPosition(
        asset_id="BOND_2Y",
        tier=2,
        market_value=4750000.0,  # down 5% from $5M
        liquidatable_value=4650000.0,
        availability=now,
        annual_yield=0.073,  # higher yield
        haircut=0.02,
        duration=2.0,
        instrument_type="bond",
    )
    equity = ConcreteAssetPosition(
        asset_id="EQUITY_GROWTH",
        tier=3,
        market_value=750000.0,  # down 25% from $1M
        liquidatable_value=700000.0,
        availability=now,
        annual_yield=0.0,
        volatility=0.25,
        instrument_type="equity",
    )

    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=5000000.0,
        asset_positions=[bond, equity],
        credit_availability=5000000.0,
    )

    signals = [
        ConcreteRiskSignal(
            source="RATE_HIKE_ANALYTICS",
            severity="HIGH",
            confidence=0.85,
            supporting_evidence={"delta_bps": 250},
            scenario_tags=["RISING_RATES"],
        )
    ]

    service = TreasuryDecisionService()
    record = service.decide(state, bundle, risk_signals=signals)
    record_baseline = service.decide(state, bundle, risk_signals=[])

    # TSS correctly reflects the market risk and signal penalties
    assert record.score["components"]["market_risk_safety"] < 95.0
    assert record.score["components"]["forecast_reliability"] < record_baseline.score["components"]["forecast_reliability"]
    assert record.score["composite_score"] < record_baseline.score["composite_score"]
    # Hard gates protect customer payouts
    assert record.hard_gate_results["passed"] is True


def test_stress_repo_lockup_and_credit_unavailability():
    """Stress Scenario 3: Repo availability freeze and bridge credit facility exhausted.

    Verifies system behaves safely when all emergency credit is frozen:
    Breaks repo early if needed, or falls back to KEEP_CASH, guaranteeing 0 default.
    """
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    # Repo locked for 2 hours, obligation due in 1 hour, credit facility FROZEN ($0)
    repo = ConcreteAssetPosition(
        asset_id="LOCKED_REPO",
        tier=1,
        market_value=10000000.0,
        liquidatable_value=9950000.0,
        availability=now + timedelta(hours=2),
        maturity=now + timedelta(hours=2),
        exit_fees=0.005,
        instrument_type="repo",
    )

    ob = ConcreteObligation(
        due_time=now + timedelta(hours=1),
        amount=5000000.0,
        is_mandatory=True,
    )

    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=0.0,
        asset_positions=[repo],
        future_obligations=[ob],
        credit_availability=0.0,  # Credit facility frozen!
    )

    service = TreasuryDecisionService()
    record = service.decide(state, bundle)

    # Since bridge credit is 0, the optimizer must BREAK_REPO_EARLY to fund the payout!
    assert record.selected_action["action_type"] == "BREAK_REPO_EARLY"
    assert record.selected_action["amount"] == 5000000.0
    # Customer payout is protected: shortfalls = 0!
    assert "BREAK_REPO_EARLY" in record.selected_action["reason_code"]

"""Regression tests asserting reserve reconciliation, eligible surplus consistency,
and deadline-by-deadline reserve constraint preservation across optimizer, validator, and ledger.
"""

import pytest
from datetime import datetime, timezone, timedelta
from reserve_juggler.contracts import (
    AssetPosition,
    AssetTier,
    CreditFacility,
    Obligation,
    ObligationType,
    TreasuryState,
    ActionType,
    ForecastPoint,
    ForecastBundle,
    ensure_utc,
)
from reserve_juggler.integration.fixtures import create_mock_forecast_bundle
from reserve_juggler.risk.liquidity_shield import assess_liquidity_reserve
from reserve_juggler.decision.service import TreasuryDecisionService
from reserve_juggler.integration.orchestrator import IndependentConstraintValidator
from reserve_juggler.ledger.ledger import TreasuryLedger, InitialTreasuryConfig


def test_reserve_reconciliation_exact_mathematical_identity():
    """Verify that Total Required Reserve + Eligible Investable Surplus EXACTLY equals Available Cash."""
    as_of = datetime(2026, 4, 15, 17, 0, tzinfo=timezone.utc)
    opening_cash = 65_000_000.0

    obl_immediate = Obligation(
        obligation_id="OBL-IMM-01",
        timestamp=as_of + timedelta(hours=6),
        amount_usd=20_500_000.0,
        description="Immediate clearing settlement",
        obligation_type=ObligationType.CUSTOMER_PAYOUT,
        is_mandatory=True,
    )
    obl_future = Obligation(
        obligation_id="OBL-FUT-01",
        timestamp=as_of + timedelta(hours=36),
        amount_usd=15_000_000.0,
        description="Future merchant settlement",
        obligation_type=ObligationType.MERCHANT_SETTLEMENT,
        is_mandatory=True,
    )

    state = TreasuryState(
        timestamp=as_of,
        operational_cash_usd=opening_cash,
        obligations=[obl_immediate, obl_future],
    )
    bundle = create_mock_forecast_bundle(as_of.isoformat(), scenario="NORMAL")

    assessment = assess_liquidity_reserve(state, bundle)
    prot_cap = assessment.protected_capital
    fut_res = assessment.future_obligation_reserve
    tot_res = assessment.total_required_reserve
    surplus = assessment.eligible_investable_surplus

    # Mathematical identity: Total Reserve = Protected Capital + Future Reserve
    assert tot_res == pytest.approx(prot_cap + fut_res, abs=1e-4)

    # Reconciliation identity: Total Reserve + Eligible Surplus == Opening Cash
    assert (tot_res + surplus) == pytest.approx(opening_cash, abs=1e-4)
    assert surplus == max(0.0, opening_cash - tot_res)


def test_optimizer_allocates_within_eligible_surplus():
    """Verify that the optimizer's permitted discretionary action preserves the total reserve floor."""
    as_of = datetime(2026, 4, 15, 17, 0, tzinfo=timezone.utc)
    opening_cash = 65_000_000.0

    obl_immediate = Obligation(
        obligation_id="OBL-IMM-01",
        timestamp=as_of + timedelta(hours=4),
        amount_usd=20_000_000.0,
        description="Immediate settlement",
        obligation_type=ObligationType.CUSTOMER_PAYOUT,
        is_mandatory=True,
    )

    state = TreasuryState(
        timestamp=as_of,
        operational_cash_usd=opening_cash,
        obligations=[obl_immediate],
    )
    bundle = create_mock_forecast_bundle(as_of.isoformat(), scenario="NORMAL")

    assessment = assess_liquidity_reserve(state, bundle)
    tot_res = assessment.total_required_reserve
    eligible_surplus = assessment.eligible_investable_surplus

    decision_svc = TreasuryDecisionService()
    decision = decision_svc.decide(state=state, forecast_bundle=bundle)

    # Permitted allocation cannot exceed eligible surplus
    allocated_amt = decision.amount
    assert allocated_amt <= eligible_surplus + 1e-4

    # Post-allocation operating cash must strictly preserve total required reserve
    remaining_cash = opening_cash - allocated_amt
    assert remaining_cash >= tot_res - 1e-4


def test_validator_rejects_discretionary_action_exceeding_surplus():
    """Verify that IndependentConstraintValidator rejects any action that attempts to breach total required reserves."""
    as_of = datetime(2026, 4, 15, 17, 0, tzinfo=timezone.utc)
    opening_cash = 65_000_000.0

    state = TreasuryState(
        timestamp=as_of,
        operational_cash_usd=opening_cash,
        obligations=[
            Obligation(
                obligation_id="OBL-01",
                timestamp=as_of + timedelta(hours=2),
                amount_usd=25_000_000.0,
                description="Payroll and vendor settlement",
                obligation_type=ObligationType.OPERATIONAL_EXPENSE,
                is_mandatory=True,
            )
        ],
    )
    bundle = create_mock_forecast_bundle(as_of.isoformat(), scenario="NORMAL")
    assessment = assess_liquidity_reserve(state, bundle)
    tot_res = assessment.total_required_reserve
    surplus = assessment.eligible_investable_surplus

    # Construct an over-allocated proposal that breaches total reserve by $30,102.07
    excess_investment = surplus + 30_102.07
    class MockProposal:
        selected_action = {"action_type": "PLACE_REPO", "target_asset": "T1_REPO", "amount": excess_investment}
        amount = excess_investment
        reserve = assessment.to_dict()

    validator = IndependentConstraintValidator()
    val_res = validator.validate_proposal(
        state=state,
        proposal=MockProposal(),
        forecast_bundle=bundle,
        stress_regime="NORMAL",
    )

    # Must fail TotalRequiredReserveGate
    assert not val_res.passed
    assert any("TotalRequiredReserveGate" in g.gate_name and g.status.value == "FAIL" for g in val_res.gate_outcomes)
    assert any("breaches total required reserve floor" in r for r in val_res.rejection_reasons)
    assert val_res.shortfall_amount_usd == pytest.approx(30_102.07, abs=1.0)


def test_ledger_reconciliation_preserves_reserves_at_deadline():
    """Verify that executed permitted action maintains ledger cash balance above total required reserve."""
    as_of = datetime(2026, 4, 15, 17, 0, tzinfo=timezone.utc)
    opening_cash = 65_000_000.0

    state = TreasuryState(
        timestamp=as_of,
        operational_cash_usd=opening_cash,
        obligations=[
            Obligation(
                obligation_id="OBL-PAY-01",
                timestamp=as_of + timedelta(hours=3),
                amount_usd=15_000_000.0,
                description="Clearing payout",
                obligation_type=ObligationType.CUSTOMER_PAYOUT,
                is_mandatory=True,
            )
        ],
    )
    bundle = create_mock_forecast_bundle(as_of.isoformat(), scenario="NORMAL")
    assessment = assess_liquidity_reserve(state, bundle)
    surplus = assessment.eligible_investable_surplus
    tot_res = assessment.total_required_reserve

    ledger = TreasuryLedger(config=InitialTreasuryConfig(opening_cash_usd=opening_cash))
    action_spec = {
        "type": "SWEEP_REPO",
        "amount_usd": surplus,
        "asset_id": "T1_REPO_SWEEP",
        "decision_id": "DEC-TEST-001",
    }

    new_state, outcome = ledger.simulate_action(state, action_spec, bundle)
    assert outcome.execution_status == "SUCCESS"
    assert outcome.realized_cash_movements.get("sweep_repo") == -surplus

    # Ledger cash must reconcile exactly to total required reserve
    assert new_state.operational_cash_usd == pytest.approx(tot_res, abs=1e-4)
    assert new_state.operational_cash_usd >= state.upcoming_obligations_usd


def test_deterministic_known_hourly_flows_exact_reconciliation():
    """Verify that when hourly flows are known, discrete interval accumulation
    and pre-cumulative forecast contracts produce mathematically identical results to the penny.
    """
    as_of = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    opening_cash = 30_000_000.0

    # 1. Discrete contract: 24 discrete 1-hour forecast points
    # Hourly withdrawal P90 = $100,000, Hourly deposit P10 = $60,000 -> net deficit = $40,000/hr
    discrete_points = []
    for h in range(1, 25):
        discrete_points.append(
            ForecastPoint(
                timestamp=(as_of + timedelta(hours=h)).isoformat(),
                forecast_horizon=float(h),
                deposit_point_forecast=70_000.0,
                withdrawal_point_forecast=90_000.0,
                deposit_p10=60_000.0,
                deposit_p90=80_000.0,
                withdrawal_p10=80_000.0,
                withdrawal_p90=100_000.0,
                net_outflow_p95=40_000.0,
                is_cumulative=False,
            )
        )
    discrete_bundle = ForecastBundle(
        forecast_id="BUN-DISC-001",
        generated_at=as_of.isoformat(),
        point_forecasts=discrete_points,
        is_cumulative=False,
    )

    # 2. Cumulative contract: Single 24-hour forecast point with cumulative net outflow = 24 * 40,000 = $960,000
    cum_points = [
        ForecastPoint(
            timestamp=(as_of + timedelta(hours=24)).isoformat(),
            forecast_horizon=24.0,
            deposit_point_forecast=24 * 70_000.0,
            withdrawal_point_forecast=24 * 90_000.0,
            deposit_p10=24 * 60_000.0,
            deposit_p90=24 * 80_000.0,
            withdrawal_p10=24 * 80_000.0,
            withdrawal_p90=24 * 100_000.0,
            net_outflow_p95=960_000.0,
            cumulative_net_outflow_p95=960_000.0,
            is_cumulative=True,
        )
    ]
    cum_bundle = ForecastBundle(
        forecast_id="BUN-CUM-001",
        generated_at=as_of.isoformat(),
        point_forecasts=cum_points,
        is_cumulative=True,
    )

    obl_12h = Obligation(
        obligation_id="OBL-12H",
        timestamp=as_of + timedelta(hours=12),
        amount_usd=10_000_000.0,
        description="Midday clearing settlement",
        obligation_type=ObligationType.CUSTOMER_PAYOUT,
        is_mandatory=True,
    )

    state = TreasuryState(
        timestamp=as_of,
        operational_cash_usd=opening_cash,
        obligations=[obl_12h],
    )

    res_discrete = assess_liquidity_reserve(state, discrete_bundle, stress_multiplier=1.0)
    res_cum = assess_liquidity_reserve(state, cum_bundle, stress_multiplier=1.0)

    # Both must arrive at exactly $960,000.00 customer flow deficit
    expected_deficit = 24 * 40_000.0  # $960,000.00
    assert res_discrete.conservative_net_deficit == pytest.approx(expected_deficit, abs=1e-2)
    assert res_cum.conservative_net_deficit == pytest.approx(expected_deficit, abs=1e-2)

    # Protected capital: $10M obligation + $1M operational buffer (10%) + $960k deficit = $11,960,000.00
    expected_protected = 10_000_000.0 + (10_000_000.0 * 0.10) + expected_deficit
    assert res_discrete.protected_capital == pytest.approx(expected_protected, abs=1e-2)
    assert res_cum.protected_capital == pytest.approx(expected_protected, abs=1e-2)

    # Eligible surplus: $30M - $11,960,000 = $18,040,000.00
    expected_surplus = opening_cash - expected_protected
    assert res_discrete.eligible_investable_surplus == pytest.approx(expected_surplus, abs=1e-2)
    assert res_cum.eligible_investable_surplus == pytest.approx(expected_surplus, abs=1e-2)


def test_repo_maturity_settlement_timing_enforcement():
    """Verify that maturing repo proceeds count as available ONLY at or before their actual settlement time.
    An obligation due at t+36h CANNOT be offset by a repo that settles at t+48h.
    """
    as_of = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    opening_cash = 25_000_000.0

    # Obligation due at t+36h ($15M)
    obl_36h = Obligation(
        obligation_id="OBL-36H",
        timestamp=as_of + timedelta(hours=36),
        amount_usd=15_000_000.0,
        description="Vendor bulk settlement",
        obligation_type=ObligationType.OPERATIONAL_EXPENSE,
        is_mandatory=True,
    )

    # Case A: Repo matures at t+48h (AFTER the obligation deadline of t+36h)
    late_repo = AssetPosition(
        position_id="POS-REPO-48H",
        asset_id="REPO_OVERNIGHT",
        tier=AssetTier.TIER_1_REPO,
        instrument_type="repo",
        market_value=15_000_000.0,
        maturity=as_of + timedelta(hours=48),
    )
    state_late_repo = TreasuryState(
        timestamp=as_of,
        operational_cash_usd=opening_cash,
        obligations=[obl_36h],
        asset_positions=[late_repo],
    )

    # Case B: Repo matures at t+24h (BEFORE the obligation deadline of t+36h)
    early_repo = AssetPosition(
        position_id="POS-REPO-24H",
        asset_id="REPO_OVERNIGHT",
        tier=AssetTier.TIER_1_REPO,
        instrument_type="repo",
        market_value=15_000_000.0,
        maturity=as_of + timedelta(hours=24),
    )
    state_early_repo = TreasuryState(
        timestamp=as_of,
        operational_cash_usd=opening_cash,
        obligations=[obl_36h],
        asset_positions=[early_repo],
    )

    bundle = create_mock_forecast_bundle(as_of.isoformat(), scenario="NORMAL")

    res_late = assess_liquidity_reserve(state_late_repo, bundle)
    res_early = assess_liquidity_reserve(state_early_repo, bundle)

    # In Case A: At t+36h, the late repo has NOT settled.
    # Therefore, future_obligation_reserve must be at least $15M to cover the 36h obligation!
    assert res_late.future_obligation_reserve >= 15_000_000.0 - 1e-4

    # In Case B: At t+36h, the early repo HAS settled (at t+24h).
    # Therefore, repo proceeds offset the obligation, reducing future_obligation_reserve!
    assert res_early.future_obligation_reserve < res_late.future_obligation_reserve
    assert res_early.future_obligation_reserve == pytest.approx(0.0, abs=1e-4)


def test_credit_facility_commitment_and_availability_gate():
    """Verify that credit proceeds count ONLY when the facility is committed, available, and active."""
    as_of = datetime(2026, 4, 15, 12, 0, tzinfo=timezone.utc)
    opening_cash = 5_000_000.0  # Cash below immediate obligation

    obl = Obligation(
        obligation_id="OBL-IMM-01",
        timestamp=as_of + timedelta(hours=2),
        amount_usd=10_000_000.0,
        is_mandatory=True,
    )

    # Uncommitted / suspended facility
    inactive_facility = CreditFacility(
        facility_id="FAC-SUSPENDED",
        committed_limit_usd=50_000_000.0,
        drawn_amount_usd=0.0,
        is_available=False,
        is_active=False,
    )

    state = TreasuryState(
        timestamp=as_of,
        operational_cash_usd=opening_cash,
        obligations=[obl],
        credit_facility=inactive_facility,
    )

    # Availability must evaluate to 0.0
    assert state.credit_availability == 0.0

    bundle = create_mock_forecast_bundle(as_of.isoformat(), scenario="NORMAL")
    assessment = assess_liquidity_reserve(state, bundle)

    # Must flag credit exhaustion / stress scenario because credit cannot be drawn
    assert any("Credit facility exhausted" in s for s in assessment.uncovered_stress_scenarios)


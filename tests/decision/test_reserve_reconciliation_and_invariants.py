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

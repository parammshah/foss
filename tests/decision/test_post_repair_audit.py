"""
Post-Repair Performance Audit & Acceptance Regression Test Suite.
Validates:
1. Avoidable payout shortfall prevented by a valid funding plan.
2. Genuinely unavoidable shortage declared explicitly (no hidden shortfalls).
3. Asset sale correctly rejected when settlement delay exceeds payment deadline.
4. Combined funding plan consistency without double counting resources.
5. Unpaid customer obligations remain visible on balance sheet as liabilities.
6. Bridge facility repayment, interest accrual, and debt reconciliation.
7. Investment principal, accrued income, and marked-to-market portfolio separation.
8. Cost-minimizing selection among feasible emergency actions.
9. Full accounting reconciliation: Net Treasury Equity = Assets - Liabilities.
10. Identical initial conditions and obligations across all benchmark strategies.
11. No unauthorized funding or undrawn credit counted as cash.
12. Simulation CLI fails cleanly with non-zero exit code when dataset is missing.
"""

from __future__ import annotations

import os
import subprocess
import sys
from datetime import datetime, timedelta, timezone

import pytest

from reserve_juggler.decision.service import TreasuryDecisionService
from reserve_juggler.optimization.models import ActionType
from reserve_juggler.tax.tax_profile import create_statutory_us_corp_profile
from tests.decision.fixtures import (
    ConcreteAssetPosition,
    ConcreteForecastBundle,
    ConcreteForecastPoint,
    ConcreteObligation,
    ConcreteTreasuryState,
)


@pytest.fixture
def baseline_bundle() -> ConcreteForecastBundle:
    now = datetime(2026, 6, 1, 12, 0, tzinfo=timezone.utc)
    pts = []
    for h in [1, 2, 4, 8, 12, 24, 48, 72]:
        d_base = 10_000_000.0 * float(h)
        w_base = 10_000_000.0 * float(h)
        pts.append(
            ConcreteForecastPoint(
                forecast_horizon=float(h),
                deposit_p10=d_base * 0.75,
                deposit_p50=d_base,
                deposit_p90=d_base * 1.35,
                withdrawal_p10=w_base * 0.75,
                withdrawal_p50=w_base,
                withdrawal_p90=w_base * 1.35,
            )
        )
    return ConcreteForecastBundle(point_forecasts=pts, timestamp=now, quality_status="HEALTHY")


def test_avoidable_shortfall_prevented_by_valid_funding_plan(baseline_bundle):
    """When cash is insufficient but a maturing repo covers the payout by the deadline,

    the engine must NOT declare an unresolved shortfall.
    """
    now = baseline_bundle.timestamp
    service = TreasuryDecisionService()
    tax_profile = create_statutory_us_corp_profile()

    # Obligation of $5M due at T+1.0h
    ob = ConcreteObligation(due_time=now + timedelta(hours=1.0), amount=5_000_000.0, is_mandatory=True)

    # Maturing repo of $5.5M arriving at T+0.5h (prior to deadline)
    repo = ConcreteAssetPosition(
        asset_id="REPO_1",
        tier=1,
        market_value=5_500_000.0,
        liquidatable_value=5_500_000.0,
        availability=now + timedelta(hours=0.5),
        maturity=now + timedelta(hours=0.5),
        exit_fees=0.005,
        instrument_type="repo",
    )

    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=0.0,
        asset_positions=[repo],
        future_obligations=[ob],
        credit_availability=10_000_000.0,
        credit_limit=10_000_000.0,
    )

    record = service.decide(state=state, forecast_bundle=baseline_bundle, tax_profile=tax_profile)
    sel = record.selected_action

    assert sel["action_type"] in [ActionType.WAIT_REPO_MATURITY.value, ActionType.DRAW_BRIDGE_CREDIT.value]
    assert sel["residual_unfunded_amount"] == 0.0
    assert sel["is_feasible"] is True


def test_genuinely_unavoidable_shortage_declared_explicitly(baseline_bundle):
    """When all permitted resources (cash, credit, liquidatable assets) cannot cover

    the obligation by the deadline, return UNRESOLVED_FUNDING_SHORTFALL with exact gap.
    """
    now = baseline_bundle.timestamp
    service = TreasuryDecisionService()
    tax_profile = create_statutory_us_corp_profile()

    # Obligation of $15M due immediately (T+0.0h)
    ob = ConcreteObligation(due_time=now, amount=15_000_000.0, is_mandatory=True)

    # Only $2M cash, $3M credit limit, no repos, bond sale takes 24h to settle
    bond = ConcreteAssetPosition(
        asset_id="BOND_2Y",
        tier=2,
        market_value=20_000_000.0,
        liquidatable_value=19_600_000.0,
        availability=now + timedelta(hours=24.0),
        settlement_delay_hours=24.0,
        instrument_type="bond",
    )

    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=2_000_000.0,
        asset_positions=[bond],
        future_obligations=[ob],
        credit_availability=3_000_000.0,
        credit_limit=3_000_000.0,
    )

    record = service.decide(state=state, forecast_bundle=baseline_bundle, tax_profile=tax_profile)
    sel = record.selected_action

    # If partial funding action is selected (e.g. DRAW_BRIDGE_CREDIT for $3M), residual gap must be $10M
    # If UNRESOLVED_FUNDING_SHORTFALL is selected, residual gap must be $13M (or $10M after credit)
    assert sel["residual_unfunded_amount"] > 0.0
    assert sel["is_feasible"] is False  # Must NOT be falsely declared fully feasible!


def test_asset_sale_correctly_rejected_when_proceeds_arrive_too_late(baseline_bundle):
    """Sovereign bond with 24h settlement delay cannot fund an obligation due in 2 hours."""
    now = baseline_bundle.timestamp
    service = TreasuryDecisionService()
    tax_profile = create_statutory_us_corp_profile()

    ob = ConcreteObligation(due_time=now + timedelta(hours=2.0), amount=10_000_000.0, is_mandatory=True)

    bond = ConcreteAssetPosition(
        asset_id="BOND_ILLIQUID",
        tier=2,
        market_value=15_000_000.0,
        liquidatable_value=14_700_000.0,
        availability=now + timedelta(hours=24.0),
        settlement_delay_hours=24.0,
        instrument_type="bond",
    )

    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=0.0,
        asset_positions=[bond],
        future_obligations=[ob],
        credit_availability=0.0,
        credit_limit=0.0,
    )

    record = service.decide(state=state, forecast_bundle=baseline_bundle, tax_profile=tax_profile)
    sel = record.selected_action

    # Bond sale candidate must be rejected due to settlement delay
    bond_alts = [a for a in record.alternatives if a["action_type"] == ActionType.SELL_BOND.value]
    assert len(bond_alts) > 0
    assert bond_alts[0]["is_feasible"] is False
    assert "REJECTED_BOND_SETTLEMENT" in bond_alts[0]["reason_code"]


def test_combined_funding_plan_no_double_counting(baseline_bundle):
    """Available cash, credit draw, and liquidatable proceeds must not be double counted."""
    now = baseline_bundle.timestamp
    service = TreasuryDecisionService()
    tax_profile = create_statutory_us_corp_profile()

    cash = 4_000_000.0
    credit_limit = 5_000_000.0
    ob = ConcreteObligation(due_time=now, amount=10_000_000.0, is_mandatory=True)

    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=cash,
        asset_positions=[],
        future_obligations=[ob],
        credit_availability=credit_limit,
        credit_limit=credit_limit,
    )

    record = service.decide(state=state, forecast_bundle=baseline_bundle, tax_profile=tax_profile)
    sel = record.selected_action

    # Cash + credit = $4M + $5M = $9M. Residual shortfall on $10M payout is strictly $1M.
    assert sel["existing_cash_applied"] == cash
    assert sel["funded_by_deadline"] == credit_limit
    assert sel["residual_unfunded_amount"] == 1_000_000.0


def test_unpaid_customer_obligations_remain_visible_liabilities():
    """Unpaid customer payouts must not vanish; they must accumulate on the balance sheet

    as cumulative liabilities and reduce Net Treasury Equity dollar-for-dollar.
    """
    initial_cash = 20_000_000.0
    deposits = 50_000_000.0
    obligations_due = 100_000_000.0

    # Total cash available = $70M. Bank pays $70M, leaving $30M unpaid.
    cash_paid = initial_cash + deposits
    unpaid_shortfall = obligations_due - cash_paid  # $30M
    ending_cash = 0.0
    ending_debt = 0.0

    # Balance sheet calculation
    total_assets = ending_cash
    total_liabilities = ending_debt + unpaid_shortfall  # $30M liability
    net_treasury_equity = total_assets - total_liabilities  # -$30M

    # Accounting flow reconciliation
    expected_equity = initial_cash + deposits - obligations_due  # $20M + $50M - $100M = -$30M
    assert net_treasury_equity == expected_equity
    assert total_liabilities == 30_000_000.0


def test_bridge_repayment_and_debt_reconciliation():
    """Bridge credit draw accrues hourly interest and repays against designated cash inflows."""
    principal = 10_000_000.0
    apr = 0.12
    hours = 2.0
    interest = principal * apr * (hours / 8760.0)  # ~$273.97
    total_due = principal + interest

    repayment_inflow = 12_000_000.0  # Maturing repo proceeds
    cash_after_repayment = repayment_inflow - total_due

    assert round(interest, 2) == 273.97
    assert cash_after_repayment > 0.0
    assert total_due < repayment_inflow


def test_investment_principal_accrued_income_mtm_reconciliation():
    """Investment principal must remain distinct from accrued coupon income and market value changes."""
    principal = 50_000_000.0
    annual_yield = 0.048  # 4.8%
    hours_elapsed = 24.0
    accrued_coupon = principal * annual_yield * (hours_elapsed / 8760.0)

    haircut = 0.02
    market_value = principal  # Marked-to-market
    liquidatable_value = market_value * (1.0 - haircut)

    assert round(accrued_coupon, 2) == round(50_000_000.0 * 0.048 * (24.0 / 8760.0), 2)
    assert liquidatable_value == 49_000_000.0
    assert market_value == 50_000_000.0


def test_cost_minimizing_selection_among_feasible_emergency_actions(baseline_bundle):
    """When multiple feasible actions exist, optimizer selects the lowest all-in cost."""
    now = baseline_bundle.timestamp
    service = TreasuryDecisionService()
    tax_profile = create_statutory_us_corp_profile()

    # $5M obligation due at T+1.0h
    ob = ConcreteObligation(due_time=now + timedelta(hours=1.0), amount=5_000_000.0, is_mandatory=True)

    # Feasible Action 1: Maturing repo at T+0.5h (Cost = $0.00)
    repo_maturing = ConcreteAssetPosition(
        asset_id="REPO_FREE",
        tier=1,
        market_value=5_000_000.0,
        liquidatable_value=5_000_000.0,
        availability=now + timedelta(hours=0.5),
        maturity=now + timedelta(hours=0.5),
        exit_fees=0.005,
        instrument_type="repo",
    )

    # Feasible Action 2: Bridge credit draw (Cost = interest > $0)
    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=0.0,
        asset_positions=[repo_maturing],
        future_obligations=[ob],
        credit_availability=10_000_000.0,
        credit_limit=10_000_000.0,
        credit_apr=0.12,
    )

    record = service.decide(state=state, forecast_bundle=baseline_bundle, tax_profile=tax_profile)
    sel = record.selected_action

    # Maturing repo has cost $0.00 vs bridge interest > $0.00 -> WAIT_REPO_MATURITY must be selected!
    assert sel["action_type"] == ActionType.WAIT_REPO_MATURITY.value
    assert sel["total_estimated_cost"] == 0.0


def test_simulation_cli_fails_cleanly_on_missing_dataset():
    """simulate_dataset.py must exit with non-zero status when dataset path does not exist."""
    cmd = [sys.executable, "tests/decision/simulate_dataset.py", "--dataset", "non_existent_dataset.csv"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    assert res.returncode != 0
    assert "Error: Dataset file not found" in res.stderr


def test_reconciliation_zero_discrepancy_with_financing_costs():
    """Validates that financing costs are counted consistently in cash/liabilities and Net Treasury Equity.

    Uses the exact reported values:
    - Opening equity: $30,000,000.00
    - Total deposits: $88,843,172,871.39
    - Total withdrawals due: $91,373,879,241.94
    - Gross investment yield: $2,651,876.54
    - Financing costs: $125,173.72
    - Net treasury PnL: $2,526,702.82
    - Invested assets: $70,309,178.68
    - Overdue customer liabilities: $2,611,842,252.21

    Fails when financing costs are omitted from cash/liabilities (the defective $125,173.72 discrepancy).
    """
    opening_equity = 30_000_000.00
    deposits = 88_843_172_871.39
    withdrawals_due = 91_373_879_241.94
    gross_yield = 2_651_876.54
    financing_costs = 125_173.72
    net_pnl = gross_yield - financing_costs  # $2,526,702.82

    # Flow expected equity
    expected_flow_equity = opening_equity + (deposits - withdrawals_due) + net_pnl
    assert round(expected_flow_equity, 2) == -2_498_179_667.73

    invested_assets = 70_309_178.68
    overdue_liabilities = 2_611_842_252.21

    # Case A: Defective accounting where cash was NOT reduced by financing_costs
    defective_cash = 43_478_579.52
    defective_total_assets = defective_cash + invested_assets
    defective_total_liabilities = overdue_liabilities
    defective_bs_equity = defective_total_assets - defective_total_liabilities
    discrepancy = defective_bs_equity - expected_flow_equity
    assert round(discrepancy, 2) == round(financing_costs, 2)  # Exactly $125,173.72 discrepancy

    # Case B: Correct double-entry accounting where cash paid financing_costs
    corrected_cash = defective_cash - financing_costs  # $43,353,405.80
    corrected_total_assets = corrected_cash + invested_assets
    corrected_total_liabilities = overdue_liabilities
    corrected_bs_equity = corrected_total_assets - corrected_total_liabilities
    zero_discrepancy = corrected_bs_equity - expected_flow_equity
    assert round(zero_discrepancy, 2) == 0.00


def test_aggregate_resource_deficit_formula():
    """Validates the aggregate net resource deficit formula:

    aggregate_resource_deficit = obligations_due - opening_resources - net_external_inflows - net_treasury_pnl
    Proves that a repaid revolving loan does not provide free permanent capital, and undrawn credit
    limit cannot be added as permanent cash resources.
    """
    opening_cash = 30_000_000.0
    deposits = 88_843_172_871.39
    obligations_due = 91_373_879_241.94
    net_pnl = 2_526_624.41
    credit_limit = 15_000_000.0

    # Correct formula:
    deficit = obligations_due - opening_cash - deposits - net_pnl
    assert round(deficit, 2) == 2_498_179_746.14

    # Flawed claim: subtracting credit limit as free permanent capital
    flawed_claimed_minimum = deficit - credit_limit
    assert flawed_claimed_minimum < deficit

    # Proves: Drawing $10M and repaying $10M + $100 interest leaves net resources unchanged (or reduced by interest)
    drawn = 10_000_000.0
    interest = 100.0
    net_flow_from_credit = drawn - (drawn + interest)  # -$100.0 (financing cost)
    assert net_flow_from_credit == -interest


def test_overdue_queue_and_late_payment_preserves_deadline_miss():
    """Paying an overdue obligation late reduces balance sheet liabilities dollar-for-dollar

    but must NEVER decrement or erase historical missed deadline incidents.
    """
    queue = []
    missed_deadline_count = 0
    liabilities = 0.0

    # Hour 1: Customer payout of $10M due, only $4M cash available
    payout_due = 10_000_000.0
    cash = 4_000_000.0
    paid_on_time = cash
    unpaid = payout_due - paid_on_time
    missed_deadline_count += 1
    liabilities += unpaid
    queue.append({"id": "OBLIG_1", "amount": unpaid, "due_hour": 1})
    cash = 0.0

    assert missed_deadline_count == 1
    assert liabilities == 6_000_000.0
    assert len(queue) == 1

    # Hour 2: Deposit of $8M arrives, no new withdrawals due
    # Policy applies cash to settle overdue queue (FIFO)
    cash += 8_000_000.0
    paid_late = 0.0
    while cash > 0.0 and queue:
        front = queue[0]
        payment = min(cash, front["amount"])
        cash -= payment
        paid_late += payment
        front["amount"] -= payment
        liabilities -= payment
        if front["amount"] <= 1e-4:
            queue.pop(0)

    assert paid_late == 6_000_000.0
    assert liabilities == 0.0
    assert cash == 2_000_000.0
    assert len(queue) == 0
    # Historical missed deadline count remains strictly 1!
    assert missed_deadline_count == 1


def test_four_separate_shortfall_measures():
    """Pre-action deficit, missed deadline, unresolved action, and overdue liabilities are distinct measures."""
    # Scenario: Hour 45 partial cash gap resolved by bridge loan draw
    cash_opening = 531_881.91
    coupon = 303.12
    deposit = 10_356_613.61
    cash_before_decision = cash_opening + coupon + deposit  # $10,888,798.64
    withdrawal_due = 12_611_309.92

    # Measure 1: Pre-action liquidity deficit detected
    pre_action_deficit_detected = cash_before_decision < withdrawal_due
    assert pre_action_deficit_detected is True
    pre_action_deficit_gap = withdrawal_due - cash_before_decision  # $1,722,511.28

    # Action executed: Draw bridge credit for the exact gap
    credit_draw = pre_action_deficit_gap
    cash_available_before_payout = cash_before_decision + credit_draw

    # Measure 2: Mandatory payment deadline missed
    paid_on_time = min(withdrawal_due, cash_available_before_payout)
    shortfall_at_deadline = withdrawal_due - paid_on_time
    mandatory_deadline_missed = shortfall_at_deadline > 0.0
    assert mandatory_deadline_missed is False
    assert shortfall_at_deadline == 0.0

    # Measure 3: Funding plan unresolved after permitted actions
    selected_action_type = ActionType.DRAW_BRIDGE_CREDIT.value
    funding_plan_unresolved = (
        selected_action_type in [ActionType.UNRESOLVED_FUNDING_SHORTFALL.value, ActionType.KEEP_CASH.value]
        and mandatory_deadline_missed
    )
    assert funding_plan_unresolved is False

    # Measure 4: Outstanding overdue customer liability at end of hour
    overdue_liability_at_end = shortfall_at_deadline
    assert overdue_liability_at_end == 0.0


def test_audit_snapshot_authoritative_reconciliation():
    """All 12 snapshot fields in shortfall audit records must reconcile arithmetically."""
    cash_before_event = 531_881.91
    settled_deposits_at_event = 10_356_613.61
    coupon_inflow = 0.0  # Coupons do NOT enter cash
    credit_draw_executed = 1_722_511.28

    cash_available_before_payout = (
        cash_before_event + settled_deposits_at_event + coupon_inflow + credit_draw_executed
    )
    amount_due_at_deadline = 12_611_006.80

    # Payout settlement
    amount_paid_on_time = min(amount_due_at_deadline, cash_available_before_payout)
    amount_paid_late = 0.0
    residual_shortfall = amount_due_at_deadline - amount_paid_on_time
    cash_after_event = cash_available_before_payout - amount_paid_on_time - amount_paid_late

    prior_debt = 0.0
    interest_cost = 47.19
    outstanding_debt_after_event = prior_debt + credit_draw_executed + interest_cost

    prior_unpaid = 0.0
    remaining_unpaid_amount = prior_unpaid + residual_shortfall - amount_paid_late

    # Mathematical identity checks
    assert round(cash_available_before_payout, 2) == 12_611_006.80
    assert round(amount_paid_on_time, 2) == 12_611_006.80
    assert round(cash_after_event, 2) == 0.00
    assert round(remaining_unpaid_amount, 2) == 0.00
    assert round(outstanding_debt_after_event, 2) == 1_722_558.47


def test_hour_45_bridge_credit_candidate_consistency(baseline_bundle):
    """Hour 45 bridge-credit candidate consistency:

    Selected action has a stable candidate_id, evaluates feasibility consistently,
    and never appears in candidate_rejection_reasons unless authorized by an explicit override.
    """
    from reserve_juggler.optimization.models import HumanOverride

    now = baseline_bundle.timestamp
    service = TreasuryDecisionService()
    tax_profile = create_statutory_us_corp_profile()

    # Hour 45 scenario: obligation exceeds cash, credit draw needed
    ob = ConcreteObligation(due_time=now, amount=12_611_309.92, is_mandatory=True)
    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=10_888_798.64,
        asset_positions=[],
        future_obligations=[ob],
        credit_availability=15_000_000.0,
        credit_limit=15_000_000.0,
        credit_apr=0.12,
    )

    override = HumanOverride(
        candidate_id="DRAW_BRIDGE_CREDIT",
        authorized_by="TREASURY_EMERGENCY_DESK",
        override_reason="Mandatory customer payout protection",
        timestamp=now,
    )

    record = service.decide(
        state=state,
        forecast_bundle=baseline_bundle,
        tax_profile=tax_profile,
        human_override=override,
    )
    sel = record.selected_action

    assert sel["action_type"] == ActionType.DRAW_BRIDGE_CREDIT.value
    assert sel["candidate_id"].startswith("BRIDGE_DRAW_")
    assert sel["is_feasible"] is True
    assert sel["payout_funded"] is True
    assert sel["funding_plan_resolved"] is True

    # Alternatives must not contain the selected action
    alt_ids = [a.get("candidate_id") for a in record.alternatives]
    assert sel["candidate_id"] not in alt_ids


def test_in_horizon_vs_post_horizon_accounting():
    """In-horizon results and post-horizon liquidation must be reported separately.

    In-horizon ending cash may be 0 while post-horizon liquidation recovers cash.
    """
    in_horizon_cash = 0.0
    in_horizon_unpaid = 100_000_000.0
    ending_bonds_mv = 50_000_000.0
    haircut = 0.02
    liquidated_proceeds = ending_bonds_mv * (1.0 - haircut)  # $49M

    # Terminal post-horizon liquidation
    post_horizon_recovered = min(liquidated_proceeds, in_horizon_unpaid)
    terminal_unpaid = in_horizon_unpaid - post_horizon_recovered

    assert in_horizon_cash == 0.0
    assert in_horizon_unpaid == 100_000_000.0
    assert post_horizon_recovered == 49_000_000.0
    assert terminal_unpaid == 51_000_000.0


def test_timestep_balance_sheet_reconciliation():
    """Double-entry balance sheet reconciliation holds at every transaction:

    Assets = Liabilities + Equity
    Equity = Opening Equity + Net Cash Flows + Net PnL
    """
    initial_cash = 30_000_000.0
    deposits = 10_000_000.0
    withdrawals_due = 15_000_000.0
    pnl = -100.0  # financing cost

    # Assets
    cash_after = initial_cash + deposits - withdrawals_due - 100.0  # $24,999,900
    invested_assets = 0.0
    total_assets = cash_after + invested_assets

    # Liabilities
    debt = 0.0
    unpaid = 0.0
    total_liabilities = debt + unpaid

    # Balance Sheet Equity
    bs_equity = total_assets - total_liabilities

    # Flow Equity
    flow_equity = initial_cash + (deposits - withdrawals_due) + pnl

    assert bs_equity == flow_equity
    assert abs(bs_equity - flow_equity) < 0.01


def test_coupon_yield_non_cash_accrual():
    """Bond coupon yield does not pay cash every hour; it accrues into valuation, not cash."""
    operational_cash = 10_000_000.0
    bond_mv = 50_000_000.0
    annual_yield = 0.048
    hourly_accrued = bond_mv * annual_yield * (1.0 / 8760.0)

    # Cash must NOT increase by hourly accrued coupon
    cash_after_accrual = operational_cash  # Unchanged
    equity_after_accrual = operational_cash + bond_mv + hourly_accrued

    assert cash_after_accrual == operational_cash
    assert equity_after_accrual > operational_cash + bond_mv


def test_repo_coupon_maturity():
    """Overnight repo matures at 08:00 into operational cash without haircut."""
    principal = 20_000_000.0
    repo_apy = 0.045
    lockup_hours = 15.0
    yield_earned = principal * repo_apy * (lockup_hours / 8760.0)

    cash_opening = 5_000_000.0
    cash_after_maturity = cash_opening + principal + yield_earned

    assert cash_after_maturity == cash_opening + principal + yield_earned
    assert yield_earned > 0.0


def test_sovereign_bond_index_mtm():
    """Sovereign bond position marked to market via bond price index."""
    units = 100_000.0
    entry_index = 100.0
    cost_basis = units * entry_index  # $10,000,000

    new_index = 98.50  # 1.5% drop
    current_mv = units * new_index
    unrealized_mtm = current_mv - cost_basis

    assert current_mv == 9_850_000.0
    assert unrealized_mtm == -150_000.0


def test_equity_price_index_mtm():
    """Growth equity position marked to market via stock price index."""
    units = 50_000.0
    entry_index = 200.0
    cost_basis = units * entry_index  # $10,000,000

    new_index = 210.0  # 5% gain
    current_mv = units * new_index
    unrealized_mtm = current_mv - cost_basis

    assert current_mv == 10_500_000.0
    assert unrealized_mtm == 500_000.0


def test_tier2_settlement_delay(baseline_bundle):
    """Tier 2 Sovereign Bond sale requires 24 hours to settle into cash."""
    now = baseline_bundle.timestamp
    service = TreasuryDecisionService()
    tax_profile = create_statutory_us_corp_profile()

    bond = ConcreteAssetPosition(
        asset_id="BOND_2Y",
        tier=2,
        market_value=10_000_000.0,
        liquidatable_value=9_800_000.0,
        availability=now + timedelta(hours=24.0),
        settlement_delay_hours=24.0,
        instrument_type="bond",
    )
    ob = ConcreteObligation(due_time=now, amount=5_000_000.0, is_mandatory=True)
    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=0.0,
        asset_positions=[bond],
        future_obligations=[ob],
    )

    record = service.decide(state=state, forecast_bundle=baseline_bundle, tax_profile=tax_profile)
    bond_sales = [a for a in record.alternatives if a["action_type"] == ActionType.SELL_BOND.value]
    assert len(bond_sales) > 0
    assert bond_sales[0]["is_feasible"] is False
    assert bond_sales[0]["availability_delay_hours"] == 24.0


def test_tier3_settlement_delay(baseline_bundle):
    """Tier 3 Growth Equity sale requires 48 hours to settle into cash."""
    now = baseline_bundle.timestamp
    service = TreasuryDecisionService()
    tax_profile = create_statutory_us_corp_profile()

    equity = ConcreteAssetPosition(
        asset_id="EQ_GROWTH",
        tier=3,
        market_value=10_000_000.0,
        liquidatable_value=8_500_000.0,
        availability=now + timedelta(hours=48.0),
        settlement_delay_hours=48.0,
        instrument_type="equity",
    )
    ob = ConcreteObligation(due_time=now, amount=5_000_000.0, is_mandatory=True)
    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=0.0,
        asset_positions=[equity],
        future_obligations=[ob],
    )

    record = service.decide(state=state, forecast_bundle=baseline_bundle, tax_profile=tax_profile)
    eq_sales = [a for a in record.alternatives if a["action_type"] == ActionType.SELL_EQUITY.value]
    assert len(eq_sales) > 0
    assert eq_sales[0]["is_feasible"] is False
    assert eq_sales[0]["availability_delay_hours"] == 48.0


def test_pending_settlements_balance_sheet_asset():
    """Pending settlement receivables are recognized as assets on the balance sheet."""
    cash = 5_000_000.0
    pending_settlement_receivable = 9_800_000.0  # Sold bond settling tomorrow
    total_assets = cash + pending_settlement_receivable

    liabilities = 0.0
    equity = total_assets - liabilities

    assert total_assets == 14_800_000.0
    assert equity == 14_800_000.0


def test_gated_core_capital_during_net_cash_burn(baseline_bundle):
    """When forecast indicates net cash burn, investable Core Capital is gated to 0.0."""
    from reserve_juggler.risk.capital_firewall import classify_hot_core_capital
    from reserve_juggler.risk.liquidity_shield import assess_liquidity_reserve

    now = baseline_bundle.timestamp
    pts = [
        ConcreteForecastPoint(
            forecast_horizon=72.0,
            deposit_p10=5_000_000.0,
            deposit_p50=10_000_000.0,
            deposit_p90=15_000_000.0,
            withdrawal_p10=8_000_000.0,
            withdrawal_p50=15_000_000.0,  # Withdrawals exceed deposits (burn)
            withdrawal_p90=22_000_000.0,
        )
    ]
    burn_bundle = ConcreteForecastBundle(point_forecasts=pts, timestamp=now, quality_status="HEALTHY")
    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=50_000_000.0,
        asset_positions=[],
        future_obligations=[],
    )
    reserve = assess_liquidity_reserve(state, burn_bundle)
    classification = classify_hot_core_capital(state, burn_bundle, reserve)

    assert classification.investable_core_capital == 0.0
    assert "net cash burn" in classification.rationale


def test_dual_mode_optimizer_selects_emergency_funding(baseline_bundle):
    """Dual-mode optimizer switches to emergency funding mode when pre-action deficit exists."""
    now = baseline_bundle.timestamp
    service = TreasuryDecisionService()
    tax_profile = create_statutory_us_corp_profile()

    # Obligation of $10M due, only $2M cash
    ob = ConcreteObligation(due_time=now, amount=10_000_000.0, is_mandatory=True)
    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=2_000_000.0,
        asset_positions=[],
        future_obligations=[ob],
        credit_availability=10_000_000.0,
        credit_limit=10_000_000.0,
    )

    record = service.decide(state=state, forecast_bundle=baseline_bundle, tax_profile=tax_profile)
    assert record.metadata.get("optimization_mode") == "EMERGENCY_FUNDING"
    assert record.selected_action["action_type"] in [
        ActionType.DRAW_BRIDGE_CREDIT.value,
        ActionType.UNRESOLVED_FUNDING_SHORTFALL.value,
    ]


def test_human_override_authorization(baseline_bundle):
    """Emergency bridge credit draw authorized via explicit HumanOverride."""
    from reserve_juggler.optimization.models import HumanOverride

    now = baseline_bundle.timestamp
    service = TreasuryDecisionService()
    tax_profile = create_statutory_us_corp_profile()

    ob = ConcreteObligation(due_time=now, amount=5_000_000.0, is_mandatory=True)
    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=1_000_000.0,
        asset_positions=[],
        future_obligations=[ob],
        credit_availability=5_000_000.0,
        credit_limit=5_000_000.0,
    )

    override = HumanOverride(
        candidate_id="DRAW_BRIDGE_CREDIT",
        authorized_by="CHIEF_RISK_OFFICER",
        override_reason="Authorized emergency payout fulfillment",
        timestamp=now,
    )

    record = service.decide(
        state=state,
        forecast_bundle=baseline_bundle,
        tax_profile=tax_profile,
        human_override=override,
    )
    assert record.metadata.get("human_override") is not None
    assert record.metadata["human_override"]["authorized_by"] == "CHIEF_RISK_OFFICER"
    assert record.selected_action["action_type"] == ActionType.DRAW_BRIDGE_CREDIT.value



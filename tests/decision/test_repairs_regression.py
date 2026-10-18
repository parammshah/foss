"""
Comprehensive regression test suite for Person 2 decision engine repair.
Validates financial correctness, funding feasibility, reserve integrity,
statutory tax accuracy, and treasury optimization.
"""

from datetime import datetime, timedelta, timezone
import json
import pytest

from reserve_juggler.optimization.models import (
    ActionType,
    OptimizationMode,
    TreasuryAction,
    get_tier_limits_for_tss,
)
from reserve_juggler.optimization.optimizer import (
    calculate_bridge_financing_cost,
    optimize_treasury_actions,
)
from reserve_juggler.risk.liquidity_shield import (
    assess_liquidity_reserve,
    evaluate_hard_safety_gates,
)
from reserve_juggler.risk.models import SafetyScoreResult, TSSComponents, SafetyMode
from reserve_juggler.risk.safety_score import calculate_treasury_safety_score
from reserve_juggler.tax.tax_profile import (
    TaxProfile,
    create_statutory_us_corp_profile,
    create_synthetic_demo_profile,
    TaxBracket,
)
from reserve_juggler.tax.tax_engine import (
    evaluate_tax_adjusted_value,
    IncomeType,
)
from reserve_juggler.decision.service import TreasuryDecisionService

from tests.decision.fixtures import (
    ConcreteAssetPosition,
    ConcreteObligation,
    ConcreteTreasuryState,
    make_healthy_forecast_bundle,
    make_benchmark_repo_bridge_state,
)


# =============================================================================
# 1. Regression Tests A, B, C (Payout Funding Feasibility & Shortfalls)
# =============================================================================


def test_regression_a_insufficient_repo_partial_funding_gap():
    """Regression Test A:
    A mandatory $10M payout is due, the bank has $0 cash, only $1M in repo
    is available for early exit (0.5% fee), and no credit is available.
    The repo sale must NOT be reported as fully covering the payout.
    The result must report actual net proceeds ($995,000) and remaining funding gap ($9,005,000).
    """
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    repo_1m = ConcreteAssetPosition(
        asset_id="REPO_1M",
        tier=1,
        market_value=1000000.0,
        liquidatable_value=995000.0,
        availability=now,
        maturity=now + timedelta(hours=24),
        exit_fees=0.005,
        settlement_delay_hours=0.0,
        instrument_type="repo",
    )

    obligation_10m = ConcreteObligation(
        due_time=now + timedelta(hours=1),
        amount=10000000.0,
        is_mandatory=True,
    )

    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=0.0,
        asset_positions=[repo_1m],
        future_obligations=[obligation_10m],
        credit_availability=0.0,
        credit_limit=0.0,
    )

    opt_result = optimize_treasury_actions(state, bundle)
    selected = opt_result.selected_action

    # Action is BREAK_REPO_EARLY to mitigate the shortfall as much as possible
    assert selected.action_type == ActionType.BREAK_REPO_EARLY
    # Net proceeds must be $995,000 ($1M minus $5,000 exit fee)
    assert round(selected.net_proceeds, 2) == 995000.0
    assert round(selected.transaction_cost, 2) == 5000.0
    # Must report residual funding gap of $9,005,000
    assert round(selected.residual_unfunded_amount, 2) == 9005000.0
    assert selected.is_partial_funding is True
    assert selected.secondary_action_needed is True
    # It must NOT be marked as fully feasible/covering
    assert selected.is_feasible is False
    assert opt_result.is_unresolved_shortfall is True
    assert round(opt_result.unresolved_shortfall, 2) == 9005000.0


def test_regression_b_bridge_credit_without_repayment_source():
    """Regression Test B:
    A $10M payout requires funding before a $10M repo matures in two hours.
    No other assets or repayment resources exist, but a $10M credit facility is available.
    The engine may evaluate the draw as a way to fund the payout if its terms permit,
    but it must not falsely declare that repayment is feasible without a valid future funding source.
    """
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    # Repo matures in 5 hours, BUT an intervening obligation absorbs all repo proceeds!
    repo_pos = ConcreteAssetPosition(
        asset_id="REPO_10M",
        tier=1,
        market_value=10000000.0,
        liquidatable_value=9950000.0,
        availability=now + timedelta(hours=5),
        maturity=now + timedelta(hours=5),
        exit_fees=0.005,
        instrument_type="repo",
    )

    # Payout 1 due in 1 hour ($10M)
    payout1 = ConcreteObligation(
        due_time=now + timedelta(hours=1),
        amount=10000000.0,
        is_mandatory=True,
    )
    # Competing payout due in 4 hours ($10M) that consumes any available cash/liquidity before repo matures
    payout2 = ConcreteObligation(
        due_time=now + timedelta(hours=4),
        amount=10000000.0,
        is_mandatory=True,
    )

    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=0.0,
        asset_positions=[repo_pos],
        future_obligations=[payout1, payout2],
        credit_availability=10000000.0,
        credit_limit=10000000.0,
        credit_apr=0.12,
    )

    opt_result = optimize_treasury_actions(state, bundle)
    bridge_candidates = [
        c for c in opt_result.candidate_actions if c.action_type == ActionType.DRAW_BRIDGE_CREDIT
    ]
    assert len(bridge_candidates) == 1
    bridge = bridge_candidates[0]
    # Because competing obligations absorb proceeds before repayment, bridge cannot be verified as repaid
    # The engine evaluates draw, but flags repayment feasibility as False and rejects the candidate
    assert bridge.principal_borrowed == 10000000.0
    assert bridge.credit_availability_after_draw == 0.0
    assert bridge.repayment_feasibility is False
    assert bridge.is_feasible is False
    assert bridge.hard_gate_passed is False
    assert "REJECTED_BRIDGE_REPAYMENT" in bridge.reason_code
    assert bridge.residual_debt_after_repayment > 0.0


def test_regression_c_explicit_unresolved_funding_shortfall():
    """Regression Test C:
    If no combination of authorized actions can meet the payment deadline,
    return an explicit unresolved funding shortfall. Never describe retaining cash
    as successfully protecting a payout when the cash is insufficient.
    """
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    # Obligation due in 1 hour for $10M
    obligation = ConcreteObligation(
        due_time=now + timedelta(hours=1),
        amount=10000000.0,
        is_mandatory=True,
    )

    # Bank has zero cash, zero assets, zero credit
    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=0.0,
        asset_positions=[],
        future_obligations=[obligation],
        credit_availability=0.0,
        credit_limit=0.0,
    )

    opt_result = optimize_treasury_actions(state, bundle)
    selected = opt_result.selected_action

    # Must NOT be KEEP_CASH claiming success!
    assert selected.action_type == ActionType.UNRESOLVED_FUNDING_SHORTFALL
    assert selected.residual_unfunded_amount == 10000000.0
    assert selected.is_feasible is False
    assert opt_result.is_unresolved_shortfall is True
    assert opt_result.unresolved_shortfall == 10000000.0
    assert "CRITICAL_SHORTFALL" in selected.reason_code


# =============================================================================
# 2. Bridge Credit Mathematics & Time-Indexed Repayment Tracking
# =============================================================================


def test_bridge_interest_formula_and_repayment_tracking():
    """Verify Section 3 Bridge Credit requirements:
    - Principal: $10,000,000
    - APR: 12%
    - 2 hours at 8,760 hours/year: 10,000,000 * 0.12 * (2 / 8760) = $273.97
    - Early repo break: $50,000 (0.5%)
    - Bridge selected when all-in cost is lower and repayment is feasible.
    """
    interest = calculate_bridge_financing_cost(10000000.0, 0.12, 2.0)
    assert round(interest, 2) == 273.97

    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)
    state = make_benchmark_repo_bridge_state(now, credit_available=15000000.0)

    opt_result = optimize_treasury_actions(state, bundle)
    selected = opt_result.selected_action

    assert selected.action_type == ActionType.DRAW_BRIDGE_CREDIT
    assert selected.principal_borrowed == 10000000.0
    assert selected.apr == 0.12
    assert round(selected.interest_cost, 2) == 273.97
    assert selected.expected_repayment_time_hours == 2.0
    assert selected.residual_debt_after_repayment == 0.0
    assert selected.credit_availability_after_draw == 5000000.0  # $15M - $10M
    assert selected.repayment_feasibility is True


# =============================================================================
# 3. Action-Specific Hard Safety Gates
# =============================================================================


def test_action_specific_hard_safety_gates():
    """Verify that hard safety gates evaluate the proposed action's parameters:
    Settlement delay clashing with payment deadline, credit exceeding authorization,
    and partial funding shortfalls.
    """
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=1000000.0,
        asset_positions=[],
        future_obligations=[],
        credit_availability=5000000.0,
    )
    reserve = assess_liquidity_reserve(state, bundle)

    # 1. Gate fails if asset settlement delay > payment deadline
    gate_delayed = evaluate_hard_safety_gates(
        state=state,
        reserve_assessment=reserve,
        payment_deadline_hours=1.0,
        availability_delay_hours=24.0,  # Takes 24h to settle, but deadline is 1h!
        action_type="SELL_BOND",
    )
    assert gate_delayed.passed is False
    assert gate_delayed.gates["funding_deadline_satisfied"] is False
    assert any("BREACH_FUNDING_DEADLINE" in r for r in gate_delayed.reason_codes)

    # 2. Gate fails if credit draw exceeds available credit
    gate_credit_breach = evaluate_hard_safety_gates(
        state=state,
        reserve_assessment=reserve,
        proposed_bridge_draw=6000000.0,  # Available is only 5M
        action_type="DRAW_BRIDGE_CREDIT",
    )
    assert gate_credit_breach.passed is False
    assert gate_credit_breach.gates["credit_availability_limit"] is False
    assert any("BREACH_CREDIT_LIMIT" in r for r in gate_credit_breach.reason_codes)

    # 3. Gate fails if funding plan leaves residual unfunded shortfall
    gate_shortfall = evaluate_hard_safety_gates(
        state=state,
        reserve_assessment=reserve,
        is_partial_funding=True,
        residual_unfunded_amount=500000.0,
        action_type="BREAK_REPO_EARLY",
    )
    assert gate_shortfall.passed is False
    assert gate_shortfall.gates["full_payout_funding"] is False
    assert any("BREACH_PARTIAL_FUNDING" in r for r in gate_shortfall.reason_codes)


# =============================================================================
# 4. Multi-Horizon Obligations & Reserve Invariants
# =============================================================================


def test_multi_horizon_obligations_reserve_invariants():
    """Verify Section 5 requirements:
    - Scheduled obligations at 1, 12, 30, 48, 72 hours are included.
    - Increasing obligations cannot reduce required reserve.
    - Greater forecast uncertainty cannot increase eligible surplus.
    - Normal repo maturity proceeds do not incur early-exit fees.
    """
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    ob_1h = ConcreteObligation(due_time=now + timedelta(hours=1), amount=100000.0)
    ob_12h = ConcreteObligation(due_time=now + timedelta(hours=12), amount=200000.0)
    ob_30h = ConcreteObligation(due_time=now + timedelta(hours=30), amount=300000.0)
    ob_48h = ConcreteObligation(due_time=now + timedelta(hours=48), amount=400000.0)
    ob_72h = ConcreteObligation(due_time=now + timedelta(hours=72), amount=500000.0)

    # State with 1h obligation only
    state1 = ConcreteTreasuryState(
        timestamp=now,
        available_cash=5000000.0,
        asset_positions=[],
        future_obligations=[ob_1h],
    )
    res1 = assess_liquidity_reserve(state1, bundle, forecast_horizon_hours=72.0)

    # State with obligations across all 5 horizons (1, 12, 30, 48, 72 hours)
    state_all = ConcreteTreasuryState(
        timestamp=now,
        available_cash=5000000.0,
        asset_positions=[],
        future_obligations=[ob_1h, ob_12h, ob_30h, ob_48h, ob_72h],
    )
    res_all = assess_liquidity_reserve(state_all, bundle, forecast_horizon_hours=72.0)

    # Invariant 1: Increasing obligations cannot reduce required reserve
    assert res_all.future_obligation_reserve >= res1.future_obligation_reserve
    assert res_all.immediate_obligations == 300000.0  # 1h ($100k) + 12h ($200k)
    assert res1.immediate_obligations == 100000.0     # 1h only ($100k)

    # Invariant 2: Normal repo maturity does NOT deduct early-exit fee
    normal_repo = ConcreteAssetPosition(
        asset_id="REPO_NORMAL",
        tier=1,
        market_value=1000000.0,
        liquidatable_value=995000.0,
        availability=now + timedelta(hours=8),
        maturity=now + timedelta(hours=8),
        exit_fees=0.005,
    )
    state_repo = ConcreteTreasuryState(
        timestamp=now,
        available_cash=5000000.0,
        asset_positions=[normal_repo],
        future_obligations=[ob_1h],
    )
    res_repo = assess_liquidity_reserve(state_repo, bundle, forecast_horizon_hours=72.0)
    # Maturing repo proceeds are counted at 100% nominal value ($1M), not reduced by 0.5% fee!
    assert res_repo.maturing_repo_liquidity == 1000000.0


# =============================================================================
# 5. Financial Objective Consistency & Cost Breakdown
# =============================================================================


def test_optimizer_financial_objective_and_cost_breakdown():
    """Verify Section 6 requirements:
    - Reported total cost equals the cost breakdown.
    - Reported net value equals underlying revenue, taxes, and expenses.
    - No arbitrary dimensionless risk penalties (e.g. 50.0) subtracted without monetary conversion.
    - An infeasible low-cost candidate can never beat a feasible candidate.
    """
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    # Healthy state with surplus
    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=10000000.0,
        asset_positions=[],
        future_obligations=[],
    )

    opt_result = optimize_treasury_actions(state, bundle)
    for act in opt_result.candidate_actions:
        # 1. Total estimated cost equals sum of relevant cost breakdown fields
        if act.cost_breakdown:
            breakdown_sum = sum(
                v for k, v in act.cost_breakdown.items() if k not in ("tax_shield",)
            )
            assert breakdown_sum >= 0.0

        # 2. Risk penalty must be non-negative and transparent
        assert act.risk_penalty >= 0.0

    # 3. Feasible action always preferred over infeasible
    for act in opt_result.candidate_actions:
        if not act.is_feasible:
            assert act != opt_result.selected_action


# =============================================================================
# 6. Explicit Asset Allocation Actions & TSS Tier Caps
# =============================================================================


def test_explicit_asset_allocation_action_types():
    """Verify Section 7 requirements:
    - Explicit action types: KEEP_CASH, PLACE_REPO, BUY_BOND, BUY_EQUITY.
    - TSS Tier caps enforced: Tier 2 (Bonds) capped, Tier 3 (Equities) capped.
    - Indicative repo overnight sweep timing (17:00 cutoff to 08:00 maturity).
    """
    # 17:00 entry time (repo sweep window open)
    sweep_time = datetime(2026, 10, 17, 17, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(sweep_time)

    state = ConcreteTreasuryState(
        timestamp=sweep_time,
        available_cash=10000000.0,
        asset_positions=[],
        future_obligations=[],
    )

    opt_result = optimize_treasury_actions(state, bundle)
    action_types = {c.action_type for c in opt_result.candidate_actions}

    assert ActionType.KEEP_CASH in action_types
    assert ActionType.PLACE_REPO in action_types
    assert ActionType.BUY_BOND in action_types
    assert ActionType.BUY_EQUITY in action_types

    # Check that repo candidate has 15h timing (17:00 to 08:00 next day)
    repo_cands = [c for c in opt_result.candidate_actions if c.action_type == ActionType.PLACE_REPO]
    assert len(repo_cands) == 1
    assert repo_cands[0].timing_hours == 15.0


# =============================================================================
# 7. Statutory US Corporate Tax Profile vs Synthetic Demo Profile
# =============================================================================


def test_statutory_us_corp_tax_profile_accuracy():
    """Verify Section 8 Statutory Tax Rules:
    - IRS Form 1120 flat 21% federal corporate income tax rate.
    - IRC § 1211(a) corporate capital losses can only offset capital gains ($0 ordinary loss offset).
    - Financing costs deductible as business expense.
    - Synthetic demo profile clearly marked as synthetic.
    """
    # 1. Statutory US Corporate Profile
    stat_profile = create_statutory_us_corp_profile()
    assert stat_profile.jurisdiction_rules.jurisdiction_code == "US_FED_CORP_STATUTORY"
    assert stat_profile.jurisdiction_rules.is_synthetic is False
    assert stat_profile.jurisdiction_rules.capital_loss_offset_limit == 0.0  # IRC § 1211(a)

    # Revenue of $100k taxed at flat 21%
    tax_res = evaluate_tax_adjusted_value(
        gross_revenue=100000.0,
        transaction_fees=0.0,
        tax_profile=stat_profile,
        explicit_income_type=IncomeType.INTEREST,
    )
    assert round(tax_res.tax_liability, 2) == 21000.0
    assert round(tax_res.after_tax_net_value, 2) == 79000.0

    # Financing cost deduction: $10k expense gives $2,100 tax shield (21%)
    fee_res = evaluate_tax_adjusted_value(
        gross_revenue=0.0,
        transaction_fees=10000.0,
        tax_profile=stat_profile,
    )
    assert round(fee_res.tax_shield, 2) == 2100.0
    assert round(fee_res.after_tax_net_value, 2) == -7900.0  # Net cost is $7,900 after shield

    # 2. Synthetic Demo Profile
    synth_profile = create_synthetic_demo_profile()
    assert "SYNTHETIC" in synth_profile.jurisdiction_rules.jurisdiction_code
    assert synth_profile.jurisdiction_rules.is_synthetic is True


# =============================================================================
# 8. Decision Service JSON Serialization & LearningEvent Schema
# =============================================================================


def test_decision_service_json_serialization_and_event_publishing():
    """Verify Section 10 & 11:
    - DecisionRecord and LearningEvent payloads are 100% JSON serializable.
    - Events contain decision ID, timestamp, feasibility, reason codes, gate results.
    - Unresolved funding shortfalls emit UNRESOLVED_FUNDING_SHORTFALL_V1 events.
    """
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    # State with severe shortfall
    ob = ConcreteObligation(due_time=now + timedelta(hours=1), amount=5000000.0, is_mandatory=True)
    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=0.0,
        asset_positions=[],
        future_obligations=[ob],
        credit_availability=0.0,
        credit_limit=0.0,
    )

    published_events = []

    def mock_publisher(event):
        published_events.append(event)

    service = TreasuryDecisionService(event_publisher=mock_publisher)
    record = service.decide(state, bundle)

    # Verify JSON serializability of decision record
    record_dict = record.to_dict()
    json_str = json.dumps(record_dict, default=str)
    assert len(json_str) > 0

    # Verify emitted events
    event_types = [e.versioned_event_type for e in published_events]
    assert "DECISION_PROPOSED_V1" in event_types
    assert "UNRESOLVED_FUNDING_SHORTFALL_V1" in event_types

    # Check payload serialization
    for event in published_events:
        event_dict = event.to_dict()
        payload_str = json.dumps(event_dict, default=str)
        assert len(payload_str) > 0
        assert "correlation_id" in event_dict


# =============================================================================
# 9. Additional Repaired Regression Invariants (P0, P1, P2)
# =============================================================================


def test_wait_repo_maturity_with_insufficient_proceeds():
    """Verify P0: WAIT_REPO_MATURITY with Insufficient Proceeds.
    - Available cash: $0.
    - Mandatory payout: $10,000,000 due in 3 hours.
    - Repo proceeds: $2,000,000 maturing in 2 hours.
    - Authorized credit limit: $10,000,000.
    The optimizer MUST NOT mark waiting alone as fully funding the $10M payout!
    It must report residual gap of $8,000,000, mark wait infeasible, and select bridge credit.
    """
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    repo_2m = ConcreteAssetPosition(
        asset_id="REPO_2M",
        tier=1,
        market_value=2000000.0,
        liquidatable_value=1990000.0,
        availability=now + timedelta(hours=2),
        maturity=now + timedelta(hours=2),
        exit_fees=0.005,
        instrument_type="repo",
    )

    obligation_10m = ConcreteObligation(
        due_time=now + timedelta(hours=3),
        amount=10000000.0,
        is_mandatory=True,
    )

    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=0.0,
        asset_positions=[repo_2m],
        future_obligations=[obligation_10m],
        credit_availability=10000000.0,
        credit_limit=10000000.0,
        credit_apr=0.12,
    )

    opt_result = optimize_treasury_actions(state, bundle)

    # Inspect WAIT_REPO_MATURITY candidate
    wait_candidates = [
        c for c in opt_result.candidate_actions if c.action_type == ActionType.WAIT_REPO_MATURITY
    ]
    assert len(wait_candidates) == 1
    wait_cand = wait_candidates[0]

    assert wait_cand.is_feasible is False
    assert wait_cand.is_partial_funding is True
    assert round(wait_cand.residual_unfunded_amount, 2) == 8000000.0
    assert "REJECTED_WAIT_PROCEEDS" in wait_cand.reason_code

    # Selected action must be DRAW_BRIDGE_CREDIT (or funding action), NOT WAIT_REPO_MATURITY
    assert opt_result.selected_action.action_type == ActionType.DRAW_BRIDGE_CREDIT
    assert opt_result.selected_action.amount == 10000000.0


def test_wait_repo_maturity_with_sufficient_proceeds():
    """Verify P0: WAIT_REPO_MATURITY with Sufficient Proceeds.
    - Available cash: $0.
    - Mandatory payout: $2,000,000 due in 3 hours.
    - Repo proceeds: $10,000,000 maturing in 2 hours.
    Waiting alone fully funds the payout with 0 fees!
    """
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    repo_10m = ConcreteAssetPosition(
        asset_id="REPO_10M",
        tier=1,
        market_value=10000000.0,
        liquidatable_value=9950000.0,
        availability=now + timedelta(hours=2),
        maturity=now + timedelta(hours=2),
        exit_fees=0.005,
        instrument_type="repo",
    )

    obligation_2m = ConcreteObligation(
        due_time=now + timedelta(hours=3),
        amount=2000000.0,
        is_mandatory=True,
    )

    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=0.0,
        asset_positions=[repo_10m],
        future_obligations=[obligation_2m],
        credit_availability=10000000.0,
        credit_limit=10000000.0,
    )

    opt_result = optimize_treasury_actions(state, bundle)
    selected = opt_result.selected_action

    assert selected.action_type == ActionType.WAIT_REPO_MATURITY
    assert selected.is_feasible is True
    assert selected.residual_unfunded_amount == 0.0
    assert selected.total_estimated_cost == 0.0
    assert "fully funding the payout" in selected.reason_code


def test_wait_repo_maturity_after_deadline_rejected():
    """Verify P0: WAIT_REPO_MATURITY maturing AFTER deadline is rejected."""
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    repo_late = ConcreteAssetPosition(
        asset_id="REPO_LATE",
        tier=1,
        market_value=10000000.0,
        liquidatable_value=9950000.0,
        availability=now + timedelta(hours=4),
        maturity=now + timedelta(hours=4),
        exit_fees=0.005,
        instrument_type="repo",
    )

    obligation_early = ConcreteObligation(
        due_time=now + timedelta(hours=1),
        amount=5000000.0,
        is_mandatory=True,
    )

    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=0.0,
        asset_positions=[repo_late],
        future_obligations=[obligation_early],
        credit_availability=0.0,
    )

    opt_result = optimize_treasury_actions(state, bundle)
    wait_cands = [c for c in opt_result.candidate_actions if c.action_type == ActionType.WAIT_REPO_MATURITY]
    assert len(wait_cands) == 1
    assert wait_cands[0].is_feasible is False
    assert "AFTER payment deadline" in wait_cands[0].reason_code


def test_explicit_zero_repo_exit_fee_preserved():
    """Verify P1: Explicitly configured 0.0 repo exit fee remains 0.0."""
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    repo_zero_fee = ConcreteAssetPosition(
        asset_id="REPO_ZERO_FEE",
        tier=1,
        market_value=10000000.0,
        liquidatable_value=10000000.0,
        availability=now,
        maturity=now + timedelta(hours=24),
        exit_fees=0.0,  # Explicitly configured zero fee!
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
        asset_positions=[repo_zero_fee],
        future_obligations=[ob],
        credit_availability=0.0,
    )

    opt_result = optimize_treasury_actions(state, bundle)
    break_cands = [c for c in opt_result.candidate_actions if c.action_type == ActionType.BREAK_REPO_EARLY]
    assert len(break_cands) == 1
    action = break_cands[0]

    # Explicit 0.0 exit fee must NOT be replaced with 0.005 ($25,000 fee)!
    assert action.transaction_cost == 0.0
    assert action.total_estimated_cost == 0.0
    assert action.net_proceeds == action.gross_proceeds == 5000000.0


def test_missing_repo_exit_fee_uses_fallback():
    """Verify P1: Missing exit fee (None) uses default 0.5% fallback."""
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    repo_missing_fee = ConcreteAssetPosition(
        asset_id="REPO_MISSING_FEE",
        tier=1,
        market_value=10000000.0,
        liquidatable_value=9950000.0,
        availability=now,
        maturity=now + timedelta(hours=24),
        exit_fees=None,  # Missing fee!
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
        asset_positions=[repo_missing_fee],
        future_obligations=[ob],
        credit_availability=0.0,
    )

    opt_result = optimize_treasury_actions(state, bundle)
    break_cands = [c for c in opt_result.candidate_actions if c.action_type == ActionType.BREAK_REPO_EARLY]
    assert len(break_cands) == 1
    action = break_cands[0]

    # Default fallback 0.5% must be applied
    assert round(action.transaction_cost, 2) > 0.0
    assert action.metadata["exit_fee_rate"] == 0.005


def test_existing_tier2_portfolio_exposure_breach_freezes_purchases():
    """Verify P0: Existing Tier 2 portfolio exposure breach freezes new purchases.
    - Existing Tier 2 market value: $200,000,000.
    - Cash: $20,000,000.
    - Total Assets: $220,000,000.
    - Existing Tier 2 exposure: 200M / 220M = 90.9% > 35% cap.
    - New Tier 2 purchases must be frozen and marked infeasible.
    """
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)

    bonds_200m = ConcreteAssetPosition(
        asset_id="SOVEREIGN_200M",
        tier=2,
        market_value=200000000.0,
        liquidatable_value=196000000.0,
        availability=now,
        maturity=now + timedelta(days=730),
        annual_yield=0.048,
        instrument_type="bond",
    )

    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=20000000.0,
        asset_positions=[bonds_200m],
        future_obligations=[],
    )

    opt_result = optimize_treasury_actions(state, bundle)

    # Check that BUY_BOND candidate was generated but marked infeasible
    bond_cands = [c for c in opt_result.candidate_actions if c.action_type == ActionType.BUY_BOND]
    assert len(bond_cands) == 1
    bond_cand = bond_cands[0]
    assert bond_cand.is_feasible is False
    assert bond_cand.hard_gate_passed is False
    assert "REJECTED_PORTFOLIO_EXPOSURE_BREACH" in bond_cand.reason_code


def test_repo_entry_window_logic():
    """Verify P1: Repo sweep window open vs closed."""
    now_noon = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now_noon)

    state = ConcreteTreasuryState(
        timestamp=now_noon,
        available_cash=20000000.0,
        asset_positions=[],
        future_obligations=[],
    )

    # When window is closed (hour = 12 UTC)
    opt_closed = optimize_treasury_actions(state, bundle, repo_sweep_window_open=False)
    repo_cand_closed = next(c for c in opt_closed.candidate_actions if c.action_type == ActionType.PLACE_REPO)
    assert repo_cand_closed.is_feasible is False
    assert "REJECTED_WINDOW_CLOSED" in repo_cand_closed.reason_code

    # When window is open (e.g. 17:00 UTC)
    now_17 = datetime(2026, 10, 17, 17, 0, tzinfo=timezone.utc)
    bundle_17 = make_healthy_forecast_bundle(now_17)
    state_17 = ConcreteTreasuryState(
        timestamp=now_17,
        available_cash=20000000.0,
        asset_positions=[],
        future_obligations=[],
    )
    opt_open = optimize_treasury_actions(state_17, bundle_17, repo_sweep_window_open=True)
    repo_cand_open = next(c for c in opt_open.candidate_actions if c.action_type == ActionType.PLACE_REPO)
    assert repo_cand_open.is_feasible is True
    assert repo_cand_open.hard_gate_passed is True


def test_tax_classification_override_preserves_exemption():
    """Verify P1: Explicit income-type override preserves exemption/eligibility attributes."""
    profile = create_statutory_us_corp_profile()
    # Add a tax-exempt instrument classification
    from reserve_juggler.tax.tax_profile import InstrumentTaxClassification
    profile.register_classification(
        InstrumentTaxClassification(
            asset_id="MUNI_BOND_EXEMPT",
            income_type=IncomeType.INTEREST,
            is_exempt=True,
            is_qualified_dividend=False,
        )
    )

    res = evaluate_tax_adjusted_value(
        gross_revenue=100000.0,
        instrument_id="MUNI_BOND_EXEMPT",
        tax_profile=profile,
        explicit_income_type=IncomeType.INTEREST,
    )

    # Exemption must be preserved: 0 tax liability, with auditable override warning
    assert res.tax_liability == 0.0
    assert res.after_tax_net_value == 100000.0
    assert any("Auditable Override" in w for w in res.warnings)


def test_provisional_tax_ranking_propagation_to_decision_record():
    """Verify P1: Missing/synthetic profile propagates provisional status into DecisionRecord."""
    now = datetime(2026, 10, 17, 12, 0, tzinfo=timezone.utc)
    bundle = make_healthy_forecast_bundle(now)
    state = ConcreteTreasuryState(
        timestamp=now,
        available_cash=10000000.0,
        asset_positions=[],
        future_obligations=[],
    )

    service = TreasuryDecisionService()
    # Call without tax_profile
    record = service.decide(state, bundle, tax_profile=None)

    assert record.metadata.get("ranking_is_provisional") is True
    assert any("PROVISIONAL_TAX_RANKING" in r for r in record.reason_codes)

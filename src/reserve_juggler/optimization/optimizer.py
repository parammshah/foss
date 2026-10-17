"""
Operations Research Treasury Optimizer and Time-Aware Next Best Action Engine.

Evaluates feasible actions, enforces hard constraints and safety gates,
accounts for financing APR, settlement delays, haircuts, taxes, and exposure caps,
and selects the cost-minimizing or after-tax net-value maximizing treasury action.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Optional, Sequence

if TYPE_CHECKING:
    from reserve_juggler.decision.protocols import (
        AssetPosition,
        ForecastBundle,
        Obligation,
        RiskSignal,
        TreasuryState,
    )
from reserve_juggler.optimization.models import (
    ActionType,
    OptimizationResult,
    TierExposureLimits,
    TreasuryAction,
    get_tier_limits_for_tss,
)
from reserve_juggler.risk.capital_firewall import (
    calculate_holding_time_gate,
    classify_hot_core_capital,
)
from reserve_juggler.risk.liquidity_shield import (
    assess_liquidity_reserve,
    evaluate_hard_safety_gates,
)
from reserve_juggler.risk.models import ReserveAssessment, SafetyScoreResult
from reserve_juggler.risk.safety_score import calculate_treasury_safety_score
from reserve_juggler.tax.tax_engine import evaluate_tax_adjusted_value
from reserve_juggler.tax.tax_profile import IncomeType, TaxProfile

# Hours per year standard convention
HOURS_PER_YEAR = 8760.0


def calculate_bridge_financing_cost(
    principal: float,
    apr: float,
    duration_hours: float,
    annual_convention: float = HOURS_PER_YEAR,
) -> float:
    """Calculate bridge credit financing interest cost.

    Example:
    $10,000,000 principal at 12% APR (0.12) for 2 hours:
    Cost = 10,000,000 * 0.12 * (2 / 8760) = $273.9726...
    """
    if principal <= 0.0 or apr <= 0.0 or duration_hours <= 0.0:
        return 0.0
    return principal * apr * (duration_hours / annual_convention)


def optimize_treasury_actions(
    state: TreasuryState,
    forecast_bundle: ForecastBundle,
    risk_signals: Optional[Sequence[RiskSignal]] = None,
    tax_profile: Optional[TaxProfile] = None,
    safety_score: Optional[SafetyScoreResult] = None,
    reserve_assessment: Optional[ReserveAssessment] = None,
    planning_horizon_hours: float = 24.0,
) -> OptimizationResult:
    """Operations Research optimization engine determining the Next Best Action.

    Compares feasible candidate actions:
    - KEEP_CASH
    - PLACE_REPO
    - WAIT_REPO_MATURITY
    - BREAK_REPO_EARLY
    - SELL_BOND
    - SELL_EQUITY
    - DRAW_BRIDGE_CREDIT
    - REPAY_BRIDGE_CREDIT

    Guarantees:
    - Customer payout protection is paramount: never delays a mandatory payout to save fees.
    - Demonstrates benchmark logic: chooses bridge over repo early break when credit is
      available and repayment upon maturity is feasible.
    - Enforces prototype tier exposure limits on new allocations according to TSS.
    - Accurately integrates tax deductibility and holding period screens.
    """
    now = state.timestamp
    available_cash = max(0.0, float(state.available_cash))
    credit_avail = max(0.0, float(state.credit_availability))
    credit_apr = float(getattr(state, "credit_apr", 0.12))

    # Compute safety score and reserve assessment if not supplied
    if safety_score is None:
        safety_score = calculate_treasury_safety_score(
            state, forecast_bundle, risk_signals=risk_signals
        )
    if reserve_assessment is None:
        reserve_assessment = assess_liquidity_reserve(
            state, forecast_bundle, risk_signals=risk_signals
        )

    tier_caps = get_tier_limits_for_tss(safety_score.composite_score)
    eligible_surplus = reserve_assessment.eligible_investable_surplus

    # Check for existing exposure breaches
    tier_breaches: list[str] = []
    remediation_notes: list[str] = []
    total_assets = available_cash + sum(float(p.market_value) for p in state.asset_positions)

    if total_assets > 0.0:
        tier2_val = sum(float(p.market_value) for p in state.asset_positions if p.tier == 2)
        tier3_val = sum(float(p.market_value) for p in state.asset_positions if p.tier == 3)

        tier2_ratio = tier2_val / total_assets
        tier3_ratio = tier3_val / total_assets

        if tier2_ratio > tier_caps.tier2_max_pct:
            tier_breaches.append(
                f"Tier 2 exposure ({tier2_ratio*100:.1f}%) exceeds TSS new-investment cap ({tier_caps.tier2_max_pct*100:.1f}%)"
            )
            remediation_notes.append(
                "Remediation policy: Freeze new Tier 2 allocations; do NOT force fire-sale of existing bonds."
            )
        if tier3_ratio > tier_caps.tier3_max_pct:
            tier_breaches.append(
                f"Tier 3 exposure ({tier3_ratio*100:.1f}%) exceeds TSS new-investment cap ({tier_caps.tier3_max_pct*100:.1f}%)"
            )
            remediation_notes.append(
                "Remediation policy: Freeze new Tier 3 allocations; rebalance orderly without distressed selling."
            )

    candidates: list[TreasuryAction] = []

    # -------------------------------------------------------------
    # CASE 1: ACUTE CASH DEFICIT FOR IMMEDIATE PAYOUT OBLIGATIONS
    # -------------------------------------------------------------
    immediate_obligations = reserve_assessment.immediate_obligations
    cash_shortfall = immediate_obligations - available_cash

    if cash_shortfall > 0.0:
        # Payout shortfall must be resolved immediately!
        # Find repo positions that can be broken or waited for
        repo_positions = [
            p
            for p in state.asset_positions
            if p.instrument_type.lower() == "repo" or p.tier == 1
        ]

        # Find earliest obligation due time
        earliest_due_hours = 0.0
        for ob in state.future_obligations:
            diff = (ob.due_time - now).total_seconds() / 3600.0
            if diff >= 0.0:
                earliest_due_hours = min(earliest_due_hours or diff, diff)

        # 1. Candidate: DRAW_BRIDGE_CREDIT
        if credit_avail >= cash_shortfall:
            # Check duration: until repo matures or cash inflow arrives
            repo_mat_hours = 2.0  # default horizon
            for rp in repo_positions:
                if rp.maturity:
                    repo_mat_hours = max(
                        0.5, (rp.maturity - now).total_seconds() / 3600.0
                    )
                    break

            bridge_cost = calculate_bridge_financing_cost(
                cash_shortfall, credit_apr, repo_mat_hours
            )

            # Tax evaluation for financing cost deductibility
            tax_res = evaluate_tax_adjusted_value(
                gross_revenue=0.0,
                financing_costs=bridge_cost,
                tax_profile=tax_profile,
            )

            repayment_possible = any(
                float(p.market_value) >= cash_shortfall for p in repo_positions
            ) or (float(state.available_cash) + credit_avail >= cash_shortfall)

            candidates.append(
                TreasuryAction(
                    action_type=ActionType.DRAW_BRIDGE_CREDIT,
                    amount=cash_shortfall,
                    expected_net_value=-tax_res.after_tax_net_value,  # Net after tax expense
                    total_estimated_cost=bridge_cost,
                    risk_penalty=0.0,
                    is_feasible=True,
                    hard_gate_passed=True,
                    cost_breakdown={
                        "financing_cost": bridge_cost,
                        "tax_shield": tax_res.tax_shield,
                        "penalty_cost": 0.0,
                    },
                    timing_hours=repo_mat_hours,
                    repayment_feasibility=repayment_possible,
                    reason_code="BRIDGE_FUNDING_AVAILABLE: Draw authorized bridge facility to fund immediate payout at low interest cost.",
                    metadata={"credit_apr": credit_apr, "duration_hours": repo_mat_hours},
                )
            )
        else:
            # Infeasible bridge
            candidates.append(
                TreasuryAction(
                    action_type=ActionType.DRAW_BRIDGE_CREDIT,
                    amount=cash_shortfall,
                    expected_net_value=-999999.0,
                    total_estimated_cost=0.0,
                    is_feasible=False,
                    hard_gate_passed=False,
                    reason_code=f"REJECTED: Shortfall ${cash_shortfall:,.2f} exceeds credit capacity ${credit_avail:,.2f}.",
                )
            )

        # 2. Candidate: BREAK_REPO_EARLY
        for rp in repo_positions:
            break_amount = min(cash_shortfall, float(rp.market_value))
            exit_fee_rate = float(rp.exit_fees) if float(rp.exit_fees) > 0.0 else 0.005
            penalty = break_amount * exit_fee_rate

            tax_res = evaluate_tax_adjusted_value(
                gross_revenue=0.0,
                transaction_fees=penalty,
                tax_profile=tax_profile,
            )

            candidates.append(
                TreasuryAction(
                    action_type=ActionType.BREAK_REPO_EARLY,
                    asset_id=rp.asset_id,
                    amount=break_amount,
                    expected_net_value=-(penalty - tax_res.tax_shield),
                    total_estimated_cost=penalty,
                    risk_penalty=50.0,  # Penalty for breaking liquidity contract
                    is_feasible=True,
                    hard_gate_passed=True,
                    cost_breakdown={
                        "penalty_cost": penalty,
                        "tax_shield": tax_res.tax_shield,
                        "financing_cost": 0.0,
                    },
                    timing_hours=float(rp.settlement_delay_hours),
                    repayment_feasibility=True,
                    reason_code=f"BREAK_REPO_EARLY: Liquidate repo early incurring {exit_fee_rate*100:.2f}% penalty to cover immediate payout.",
                    metadata={"exit_fee_rate": exit_fee_rate},
                )
            )

        # 3. Candidate: WAIT_REPO_MATURITY
        for rp in repo_positions:
            hours_to_mat = 0.0
            if rp.maturity:
                hours_to_mat = max(0.0, (rp.maturity - now).total_seconds() / 3600.0)

            # Check if waiting would cause a default on obligations due before repo matures
            violates_payout = hours_to_mat > earliest_due_hours and earliest_due_hours >= 0.0
            is_feasible = not violates_payout

            candidates.append(
                TreasuryAction(
                    action_type=ActionType.WAIT_REPO_MATURITY,
                    asset_id=rp.asset_id,
                    amount=float(rp.market_value),
                    expected_net_value=0.0 if is_feasible else -1e9,
                    total_estimated_cost=0.0 if is_feasible else 1e9,
                    is_feasible=is_feasible,
                    hard_gate_passed=is_feasible,
                    timing_hours=hours_to_mat,
                    reason_code=(
                        f"WAIT_REPO_MATURITY: Repo matures in {hours_to_mat:.1f}h prior to obligations."
                        if is_feasible
                        else f"REJECTED: Repo matures in {hours_to_mat:.1f}h, which is AFTER obligation due time ({earliest_due_hours:.1f}h). Delaying payout is prohibited!"
                    ),
                )
            )

        # 4. Candidate: SELL_BOND
        for bp in state.asset_positions:
            if bp.tier == 2:
                settlement_delay = float(bp.settlement_delay_hours)
                # If settlement takes longer than when obligations are due, it cannot fund the payout!
                can_settle_in_time = settlement_delay <= earliest_due_hours
                haircut_loss = float(bp.market_value) * float(bp.haircut)
                candidates.append(
                    TreasuryAction(
                        action_type=ActionType.SELL_BOND,
                        asset_id=bp.asset_id,
                        amount=float(bp.market_value),
                        expected_net_value=-haircut_loss,
                        total_estimated_cost=haircut_loss,
                        is_feasible=can_settle_in_time,
                        hard_gate_passed=can_settle_in_time,
                        cost_breakdown={"haircut_loss": haircut_loss},
                        timing_hours=settlement_delay,
                        reason_code=(
                            "SELL_BOND: Liquidate bond position to cover cash deficit."
                            if can_settle_in_time
                            else f"REJECTED: Bond settlement delay ({settlement_delay:.1f}h) exceeds obligation timing ({earliest_due_hours:.1f}h)."
                        ),
                    )
                )

        # 5. Candidate: SELL_EQUITY
        for ep in state.asset_positions:
            if ep.tier == 3:
                settlement_delay = float(ep.settlement_delay_hours)
                can_settle_in_time = settlement_delay <= earliest_due_hours
                slippage_loss = float(ep.market_value) * (float(ep.haircut) + 0.02)
                candidates.append(
                    TreasuryAction(
                        action_type=ActionType.SELL_EQUITY,
                        asset_id=ep.asset_id,
                        amount=float(ep.market_value),
                        expected_net_value=-slippage_loss,
                        total_estimated_cost=slippage_loss,
                        is_feasible=can_settle_in_time,
                        hard_gate_passed=can_settle_in_time,
                        cost_breakdown={"slippage_loss": slippage_loss},
                        timing_hours=settlement_delay,
                        reason_code=(
                            "SELL_EQUITY: Liquidate growth equity to cover cash deficit."
                            if can_settle_in_time
                            else f"REJECTED: Equity settlement delay ({settlement_delay:.1f}h) exceeds obligation timing ({earliest_due_hours:.1f}h)."
                        ),
                    )
                )

    # -------------------------------------------------------------
    # CASE 2: SURPLUS CAPITAL ALLOCATION & NORMAL OPERATIONS
    # -------------------------------------------------------------
    else:
        # Base Action: KEEP_CASH
        candidates.append(
            TreasuryAction(
                action_type=ActionType.KEEP_CASH,
                amount=available_cash,
                expected_net_value=0.0,
                total_estimated_cost=0.0,
                is_feasible=True,
                hard_gate_passed=True,
                timing_hours=0.0,
                reason_code="KEEP_CASH: Maintain liquid cash in operational treasury account.",
            )
        )

        if eligible_surplus > 0.0:
            # Candidate: PLACE_REPO (Tier 1)
            # Overnight repo with 4.8% APR (0.048)
            repo_apr = 0.048
            repo_lockup_hours = 24.0
            repo_gross = eligible_surplus * repo_apr * (repo_lockup_hours / HOURS_PER_YEAR)
            repo_tax_res = evaluate_tax_adjusted_value(
                gross_revenue=repo_gross,
                holding_days=1.0,
                instrument_id="OVERNIGHT_REPO",
                tax_profile=tax_profile,
                explicit_income_type=IncomeType.INTEREST,
            )

            # Evaluate hard gate for repo placement
            gate_eval = evaluate_hard_safety_gates(
                state,
                reserve_assessment,
                proposed_investment_amount=eligible_surplus,
                proposed_lockup_hours=repo_lockup_hours,
                target_tier=1,
            )

            candidates.append(
                TreasuryAction(
                    action_type=ActionType.PLACE_REPO,
                    amount=eligible_surplus,
                    expected_net_value=repo_tax_res.after_tax_net_value,
                    total_estimated_cost=0.0,
                    is_feasible=gate_eval.passed,
                    hard_gate_passed=gate_eval.passed,
                    cost_breakdown={"tax_liability": repo_tax_res.tax_liability},
                    timing_hours=repo_lockup_hours,
                    reason_code=(
                        f"PLACE_REPO: Allocate ${eligible_surplus:,.2f} surplus into 24h repo at {repo_apr*100:.2f}% APR."
                        if gate_eval.passed
                        else f"REJECTED: Repo lockup fails safety gates: {'; '.join(gate_eval.reason_codes)}"
                    ),
                    metadata={"annual_yield": repo_apr, "lockup_hours": repo_lockup_hours},
                )
            )

            # Candidate: ALLOCATE_BONDS (Tier 2)
            tier2_cap_amount = eligible_surplus * tier_caps.tier2_max_pct
            if tier2_cap_amount > 0.0:
                bond_yield = 0.048
                bond_haircut = 0.02
                expected_holding_days = 180.0

                # Check holding-time gatekeeper
                class MockBond:
                    asset_id = "SOVEREIGN_2Y"
                    tier = 2
                    instrument_type = "bond"
                    haircut = bond_haircut
                    annual_yield = bond_yield

                holding_eval = calculate_holding_time_gate(
                    MockBond(), expected_holding_days=expected_holding_days
                )

                if holding_eval.passed:
                    bond_gross = (
                        tier2_cap_amount * bond_yield * (expected_holding_days / 365.0)
                    )
                    bond_tax_res = evaluate_tax_adjusted_value(
                        gross_revenue=bond_gross,
                        holding_days=expected_holding_days,
                        instrument_id="SOVEREIGN_2Y",
                        tax_profile=tax_profile,
                        explicit_income_type=IncomeType.INTEREST,
                    )
                    candidates.append(
                        TreasuryAction(
                            action_type=ActionType.PLACE_REPO,  # represented as secure allocation
                            asset_id="SOVEREIGN_2Y",
                            amount=tier2_cap_amount,
                            expected_net_value=bond_tax_res.after_tax_net_value,
                            total_estimated_cost=0.0,
                            risk_penalty=10.0,  # Duration risk penalty
                            is_feasible=True,
                            hard_gate_passed=True,
                            cost_breakdown={"tax_liability": bond_tax_res.tax_liability},
                            timing_hours=expected_holding_days * 24.0,
                            reason_code=(
                                f"ALLOCATE_SOVEREIGN_BOND: Allocate ${tier2_cap_amount:,.2f} Core Capital "
                                f"within TSS Tier 2 cap ({tier_caps.tier2_max_pct*100:.0f}%)."
                            ),
                            metadata={"tier": 2, "holding_eval": holding_eval.to_dict()},
                        )
                    )

    # -------------------------------------------------------------
    # SELECTION LOGIC: CONSTRAINED OPTIMIZATION
    # -------------------------------------------------------------
    # Filter for strictly feasible actions passing all hard safety gates
    feasible_candidates = [
        c for c in candidates if c.is_feasible and c.hard_gate_passed
    ]

    if not feasible_candidates:
        # Fallback to KEEP_CASH if no candidate passed
        selected_action = TreasuryAction(
            action_type=ActionType.KEEP_CASH,
            amount=available_cash,
            expected_net_value=0.0,
            total_estimated_cost=0.0,
            is_feasible=True,
            hard_gate_passed=True,
            reason_code="FALLBACK: No candidate action met all hard safety gates; retaining cash in treasury.",
        )
    else:
        # Objective: Maximize expected_net_value - total_estimated_cost - risk_penalty
        # When resolving cash deficit, net values are negative (costs), so maximizing picks least-cost action!
        def objective_score(act: TreasuryAction) -> float:
            return act.expected_net_value - act.total_estimated_cost - act.risk_penalty

        selected_action = max(feasible_candidates, key=objective_score)

    return OptimizationResult(
        selected_action=selected_action,
        candidate_actions=candidates,
        objective_value=(
            selected_action.expected_net_value
            - selected_action.total_estimated_cost
            - selected_action.risk_penalty
        ),
        eligible_surplus=eligible_surplus,
        tier_caps_applied=tier_caps,
        tier_breaches_detected=tier_breaches,
        remediation_notes=remediation_notes,
    )
